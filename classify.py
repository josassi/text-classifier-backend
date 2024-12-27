import pandas as pd
import logging
from tqdm.notebook import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import json
import re
import time

logging.basicConfig(level=logging.INFO)


@dataclass
class VerbatimClassification:
    """
    Data class to store classification details for a verbatim at a specific layer.
    """
    layer: int
    category_uuid: str            # UUID of the category
    category_name: str            # Name of the category
    summary: str
    sentiment: str                # Sentiment field
    flag: str = ""
    parent_uuid: Optional[str] = None  # UUID of the parent category

def process_all_verbatims(
    client,
    df_verbatims: pd.DataFrame,
    df_categories: pd.DataFrame,
    df_relationships: pd.DataFrame,
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int,
    max_trials: int
) -> pd.DataFrame:
    """
    Process all verbatims in the dataset.
    """
    all_classifications = []

    for _, row in tqdm(df_verbatims.iterrows(), total=len(df_verbatims), desc="Processing Verbatims"):
        index = row['verbatim_index']
        verbatim = row['verbatim']
        survey = row.get('survey')
        classifications = process_verbatim_layers(
            client,
            verbatim,
            survey,
            df_categories,
            df_relationships,
            layer=1,
            parent_classification=None,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            rate_limit_sleep=rate_limit_sleep,
            max_retries=max_retries,
            max_trials=max_trials
        )
        # Store the results
        for layer, layer_classifications in classifications.items():
            for classification in layer_classifications:
                result = {
                    'verbatim_index': index,
                    'verbatim': verbatim,
                    'survey': survey,
                    'layer': layer,
                    'category_uuid': classification.category_uuid,
                    'summary': classification.summary,
                    'sentiment': classification.sentiment,
                    'flag': classification.flag
                }
                all_classifications.append(result)

    # Convert to DataFrame
    df_results = pd.DataFrame(all_classifications)
    return df_results

def process_verbatim_layers(
    client,
    verbatim: str,
    survey: str,
    df_categories: pd.DataFrame,
    df_relationships: pd.DataFrame,
    layer: int,
    parent_classification: Optional[VerbatimClassification],
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int,
    max_trials: int
) -> Dict[int, List[VerbatimClassification]]:
    """
    Recursively process a single verbatim across category layers, starting from a given parent classification.
    """
    classifications_per_layer = {}

    # Get categories for the current layer based on the parent classification
    current_layer_categories_df = get_categories_for_layer(df_categories, df_relationships, parent_classification)
    if current_layer_categories_df.empty:
        logging.info(f"No categories for layer {layer}, stopping.")
        return classifications_per_layer

    # Process the verbatim at the current layer
    current_classifications = process_verbatim_layer(
        client,
        verbatim,
        survey,
        current_layer_categories_df,
        parent_classification,
        layer,
        model_name,
        temperature,
        api_key,
        rate_limit_sleep,
        max_retries,
        max_trials
    )
    if not current_classifications:
        logging.info(f"No classifications for verbatim at layer {layer}")
        return classifications_per_layer

    classifications_per_layer[layer] = current_classifications

    # For each classification at the current layer, recursively process the next layer
    for classification in current_classifications:
        # Check if the flag indicates that we should not proceed further
        if classification.flag in ["General comment", "No comment", "Unknown classification"]:
            logging.info(f"Classification '{classification.category_name}' has flag '{classification.flag}', not proceeding to further layers.")
            continue  # Skip recursive call for this classification

        child_classifications_per_layer = process_verbatim_layers(
            client,
            verbatim,
            survey,
            df_categories,
            df_relationships,
            layer + 1,
            classification,
            model_name,
            temperature,
            api_key,
            rate_limit_sleep,
            max_retries,
            max_trials
        )
        # Merge the classifications
        for child_layer, child_classifications in child_classifications_per_layer.items():
            if child_layer in classifications_per_layer:
                classifications_per_layer[child_layer].extend(child_classifications)
            else:
                classifications_per_layer[child_layer] = child_classifications

    return classifications_per_layer

def process_verbatim_layer(
    client,
    verbatim: str,
    survey: str,
    current_layer_categories_df: pd.DataFrame,
    parent_classification: Optional[VerbatimClassification],
    layer: int,
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int,
    max_trials: int
) -> List[VerbatimClassification]:
    """
    Process a single verbatim at a specific category layer.
    """
    # Get relevant categories with summaries from LLM
    relevant_categories_summaries = get_relevant_categories_with_summaries(
        client,
        verbatim,
        survey,
        current_layer_categories_df,
        parent_classification,
        model_name,
        temperature,
        api_key,
        rate_limit_sleep,
        max_retries
    )
    current_classifications = []

    for category_name, summary in relevant_categories_summaries.items():
        exists, category_uuid, category_description = check_category_exists(category_name, current_layer_categories_df)
        flag = ""
        sentiment = "Neutral"
        if not exists:
            flag = 'New category or error'
            logging.warning(f"Category '{category_name}' does not exist in current layer")
            category_description = ""

            # Attempt to correct or propose new category
            for trial in range(max_trials):
                corrected_category_name = correct_or_propose_category(
                    client,
                    verbatim,
                    summary,
                    category_name,
                    current_layer_categories_df,
                    model_name,
                    temperature,
                    api_key,
                    rate_limit_sleep,
                    max_retries
                )
                logging.info(f"Trial {trial+1}: Received corrected/proposed category '{corrected_category_name}'")
                # Update the category
                category_name = corrected_category_name

                exists, category_uuid, category_description = check_category_exists(category_name, current_layer_categories_df)
                if exists:
                    # Validate classification with the corrected category
                    classification_result, sentiment = validate_classification_and_get_sentiment(
                        client,
                        verbatim,
                        summary,
                        category_name,
                        category_description,
                        model_name,
                        temperature,
                        api_key,
                        rate_limit_sleep,
                        max_retries
                    )
                    if classification_result != category_name:
                        flag = classification_result
                        logging.info(f"Reclassified summary '{summary}' as '{classification_result}' instead of '{category_name}'")
                    else:
                        flag = ''
                    break  # Exit the trial loop if category is valid
                else:
                    flag = 'New category or error'
                    logging.warning(f"Category '{category_name}' does not exist in current layer after trial {trial+1}")

            # If after trials the category is still invalid, keep the last proposed category
            # and set the flag accordingly
        else:
            # Validate classification and get sentiment
            classification_result, sentiment = validate_classification_and_get_sentiment(
                client,
                verbatim,
                summary,
                category_name,
                category_description,
                model_name,
                temperature,
                api_key,
                rate_limit_sleep,
                max_retries
            )
            if classification_result != category_name:
                flag = classification_result
                logging.info(f"Reclassified summary '{summary}' as '{classification_result}' instead of '{category_name}'")
            else:
                flag = ''

        classification = VerbatimClassification(
            layer=layer,
            category_uuid=category_uuid,
            category_name=category_name,
            summary=summary,
            sentiment=sentiment,
            flag=flag,
            parent_uuid=parent_classification.category_uuid if parent_classification else None
        )
        current_classifications.append(classification)

    return current_classifications

def correct_or_propose_category(
    client,
    verbatim: str,
    summary: str,
    proposed_category_name: str,
    current_layer_categories_df: pd.DataFrame,
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int
) -> str:
    """
    Ask the LLM to correct the proposed category or propose a new category if it doesn't fit.
    """
    # Get the list of valid categories
    valid_categories = current_layer_categories_df['name'].tolist()
    valid_categories_str = "\n".join(f"- {category_name}" for category_name in valid_categories)

    prompt = f"""The category you provided "{proposed_category_name}" does not match any of our existing categories for this layer.

Here is the list of valid categories:

{valid_categories_str}

Based on the following customer's response and your summary, please either:

- Correct your proposed category to match one of the valid categories (provide the exact wording), or
- If your proposed category really doesn't fit in any of the valid categories, propose a new category name.

Return the category name as a single string.

Customer's response: "{verbatim}"
Your summary: "{summary}"
"""

    response = invoke_llm(
        client,
        None,
        prompt,
        model_name,
        temperature,
        api_key,
        rate_limit_sleep,
        max_retries
    )

    try:
        corrected_category_name = response.strip().split('\n')[0].strip('"').strip()
    except:
        logging.warning("Error parsing LLM response")
        corrected_category_name = "Error"
    return corrected_category_name

def get_relevant_categories_with_summaries(
    client,
    verbatim: str,
    survey: str,
    current_layer_categories_df: pd.DataFrame,
    parent_classification: Optional[VerbatimClassification],
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int
) -> Dict[str, str]:
    """
    Use the LLM to extract relevant categories and summaries from the verbatim.
    """
    # Construct prompt
    system_prompt, prompt = construct_llm_prompt(verbatim, survey, current_layer_categories_df, parent_classification)
    # Invoke LLM
    response = invoke_llm(client, system_prompt, prompt, model_name, temperature, api_key, rate_limit_sleep, max_retries, True)
    # Parse response
    relevant_categories_summaries = parse_llm_response(response)
    return relevant_categories_summaries

def construct_llm_prompt(
    verbatim: str,
    survey: str,
    current_layer_categories_df: pd.DataFrame,
    parent_classification: Optional[VerbatimClassification]
) -> Tuple[str, str]:
    """
    Construct the system prompt and user prompt to be sent to the LLM.
    """
    # Construct categories with descriptions
    category_list_str = ''
    for idx, row in current_layer_categories_df.iterrows():
        category_list_str += f"- {row['name']}: {row['description']}\n"

    system_prompt = f"""You are an assistant that classifies customer feedback into predefined categories.

Categories for classification at the current layer:
{category_list_str}

"""

    if parent_classification is not None:
        system_prompt += "Parent classification:\n"
        system_prompt += f"- Category: {parent_classification.category_name}, Summary: {parent_classification.summary}, Sentiment: {parent_classification.sentiment}\n"
        system_prompt += "\n"

    system_prompt += """Please analyze the customer's response and identify any categories from the current layer that are relevant to the response. For each relevant category, provide a brief English summary (one sentence) of the part of the response related to the category. Do not extrapolate in the summary.

Provide your answer in the following JSON format:

{
  "Category 1": "English summary of the answer that falls under 'Category 1'.",
  "Category 2": "English summary of the answer that falls under 'Category 2'",
  ...
}

If no categories from the current layer are relevant, return an empty JSON object: {}

Do not include any additional text outside the JSON object.
"""

    system_prompt += """
IMPORTANT:
- ***Return at least 1 category, even if the comment is very general*** 
"""

    prompt = f"""
Current survey: "{survey}"    
Customer's response: "{verbatim}"
"""

    return system_prompt, prompt

def invoke_llm(
    client,
    system_prompt: Optional[str],
    prompt: str,
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int,
    json_type=False,
    max_tokens: int = None,
) -> str:
    """
    Interface with the LLM API to get responses based on the provided prompts.
    """
    retries = 0

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    while retries < max_retries:
        try:
            if json_type:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    seed=42,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    seed=42,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if 'Error code: 400' in error_message and 'content_filter' in error_message:
                logging.error(f"LLM request failed due to content filtering: {e}")
                # Do not retry
                break
            else:
                logging.error(f"Error invoking LLM: {e}")
                logging.info(f"Sleeping for {rate_limit_sleep} seconds before retrying...")
                time.sleep(rate_limit_sleep)
                retries += 1
                continue
    logging.error("Max retries reached or content filter error. Returning empty string.")
    return ""

def parse_llm_response(response_text: str) -> Dict[str, str]:
    """
    Parse the LLM's response text to extract categories and summaries.
    """
    try:
        # Use regex to extract JSON part of the response
        if response_text:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_text = match.group(0)
                parsed_response = json.loads(json_text)
                if isinstance(parsed_response, dict):
                    return parsed_response
                else:
                    logging.error("Parsed response is not a dictionary")
                    return {}
            else:
                logging.error("No JSON object found in the LLM response")
                return {}
        else:
            logging.warning(f"LLM response is None")
            return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM response: {e}")
        return {}

def validate_classification_and_get_sentiment(
    client,
    verbatim: str,
    summary: str,
    category_name: str,
    category_description: str,
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int
) -> Tuple[str, str]:
    """
    Validate whether the summary truly fits the given category and extract the sentiment.

    Returns:
        Tuple[str, str]: A tuple containing the classification result (category) and sentiment.
    """
    prompt = f"""We need to classify the following verbatim and determine the sentiment (Positive, Neutral, Negative) associated with it.

Original verbatim: "{verbatim}"
Proposed summary related to the category: "{summary}"

Please perform the following:

1. Classify the verbatim into one of these categories and return it as the classification result. The category MUST be one of ["{category_name}", "General comment", "No comment"]:
   - "{category_name}" if the verbatim is explicitly related to this category description: "{category_description}"
   - "General comment" if the verbatim is too short or general to explicitly mention the above category
   - "No comment" if the verbatim is just a remark or not relevant for an analysis

2. Determine the sentiment associated with the category. ***If the category does not concern all the text, only return the sentiment related to the category independently***. The sentiment should be one of:
   - "Positive"
   - "Neutral"
   - "Negative"

Return your answer in the following JSON format:

{{
  "classification_result": "Category Name",
  "sentiment": "Sentiment from Step 2"
}}

Do not include any additional text outside the JSON object.
"""
    response = invoke_llm(
        client,
        None,
        prompt,
        model_name,
        temperature,
        api_key,
        rate_limit_sleep,
        max_retries,
        json_type=True
    )

    # Parse the JSON response
    try:
        parsed_response = json.loads(response)
        classification_result = parsed_response.get("classification_result", "").strip()
        sentiment = parsed_response.get("sentiment", "").strip()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM response: {e}")
        # Fallback to default values if parsing fails
        classification_result = "Unknown classification"
        sentiment = "Neutral"

    # Validate classification result
    valid_options = [category_name, "General comment", "No comment"]
    if classification_result not in valid_options:
        logging.warning(f"Unexpected classification result: {classification_result}")
        classification_result = "Unknown classification"

    # Validate sentiment
    valid_sentiments = ["Positive", "Neutral", "Negative"]
    if sentiment not in valid_sentiments:
        logging.warning(f"Unexpected sentiment: {sentiment}")
        sentiment = "Neutral"

    return classification_result, sentiment

def get_categories_for_layer(
    df_categories: pd.DataFrame,
    df_relationships: pd.DataFrame,
    parent_classification: Optional[VerbatimClassification]
) -> pd.DataFrame:
    """
    Retrieve categories and their descriptions for the specified layer, based on parent classification.
    """
    if parent_classification is None:
        # At the root layer (layer 1), directly retrieve categories that do not have a parent in df_relationships
        root_uuids = df_categories[~df_categories['uuid'].isin(df_relationships['child_uuid'])]['uuid'].unique()
    else:
        # Get child categories where parent_uuid matches the category_uuid of the parent classification
        parent_uuid = parent_classification.category_uuid
        child_relationships = df_relationships[df_relationships['parent_uuid'] == parent_uuid]
        root_uuids = child_relationships['child_uuid'].unique()

    # Get the corresponding categories from df_categories
    current_categories_df = df_categories[df_categories['uuid'].isin(root_uuids)][['uuid', 'name', 'description']]
    return current_categories_df

def check_category_exists(
    category_name: str,
    current_layer_categories_df: pd.DataFrame
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if the category exists in the current layer's category DataFrame and retrieve its uuid and description.
    """
    match = current_layer_categories_df[current_layer_categories_df['name'] == category_name]
    if not match.empty:
        category_uuid = match.iloc[0]['uuid']
        category_description = match.iloc[0]['description']
        return True, category_uuid, category_description
    else:
        # Return the specified UUID when category does not exist
        return False, "1cc69760-a513-4e09-9398-2d0ac16685d1", None
    



def process_verbatims_in_batches(
    client,
    df_verbatims: pd.DataFrame,
    df_categories: pd.DataFrame,
    df_relationships: pd.DataFrame,
    max_layers: int,
    model_name: str,
    temperature: float,
    api_key: str,
    rate_limit_sleep: int,
    max_retries: int,
    max_trials: int,
    batch_size: int,
    write_df_result: Callable[[pd.DataFrame, str], None],
    path_template: str,
    schema
):
    """
    Process verbatims in batches and output the results after processing every x verbatims.

    Args:
        client: The OpenAI API client instance.
        df_verbatims (pd.DataFrame): DataFrame containing verbatims to process.
        df_categories (pd.DataFrame): DataFrame containing categories names and descriptions.
        df_relationships (pd.DataFrame): DataFrame containing relationship between categories.
        model_name (str): OpenAI model name to use for the LLM.
        temperature (float): Temperature setting for the LLM.
        api_key (str): OpenAI API key.
        rate_limit_sleep (int): Time to wait (in seconds) after a rate limit error before retrying.
        max_retries (int): Maximum number of times to retry after an exception.
        max_trials (int): Maximum number of trials to correct or propose a category.
        batch_size (int): Number of verbatims to process per batch.
        write_df_result (Callable[[pd.DataFrame, str], None]): Function to write the batch results.
        path_template (str): Template string for the output path, e.g., 'results_batch_{batch_number}.csv'.
        schema: The spark schema of the output.

    """
    total_verbatims = len(df_verbatims)
    num_batches = (total_verbatims + batch_size - 1) // batch_size  # Ceiling division

    for batch_number in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = batch_number * batch_size
        end_idx = min((batch_number + 1) * batch_size, total_verbatims)
        batch_df_verbatims = df_verbatims.iloc[start_idx:end_idx].copy()
        
        logging.info(f"Processing batch {batch_number + 1}/{num_batches} with verbatims {start_idx} to {end_idx - 1}")
        
        df_results = process_all_verbatims(
            client,
            batch_df_verbatims,
            df_categories,
            df_relationships,
            model_name,
            temperature,
            api_key,
            rate_limit_sleep,
            max_retries,
            max_trials
        )
        # Generate the path for the current batch
        path = path_template.format(batch_number=batch_number + 1)  # Adding 1 to make it 1-based indexing
        logging.info(f"Batch {batch_number + 1} results has lenght: {len(df_results)}")
        if len(df_results)>0:
            try:
                write_df_result(df_results[["verbatim_index","verbatim","survey","layer","summary","flag","sentiment","category_uuid"]], path, schema)
                logging.info(f"Batch {batch_number + 1} results written to {path}")
            except Exception as e:
                logging.error(f"Batch {batch_number + 1} resulted in an error when writing the file: {e}")
                return df_results
        else:
            logging.warning(f"Batch {batch_number + 1} result is empty.")