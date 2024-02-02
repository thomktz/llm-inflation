from ecb_scraper import load_ecb_conferences
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

model = "gpt-3.5-turbo-1106"
categories = [
    "Interest rates", 
    "Employment", 
    "Energy prices", 
    "Inflation expectations", 
    "GDP growth", 
    "Supply Chain Conditions", 
    "Wage Growth"
]

def system_(date, categories):
    return f"""
        Reset context. Analyze the following ECB press conference from {date} without using any knowledge beyond that date. 
        Provide a list of sentiment scores for each of the specified categories using a scale from 0 to 9, where 0 indicates a future decrease, 4 or 5 indicates no change, and 9 indicates a future increase.
        Categories: {", ".join(categories)}
        The format for the output should be exactly 7 integers, comma-separated, corresponding to each category in the order listed above.
        An example output might look like this: 5,4,4,9,0,4,7
        Please adhere to this format strictly.
    """
    
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion(client, user, date, categories, model=model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_(date, categories)},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        # Just return the response for now, handle exceptions in a uniform way
        return response
    except Exception as e:
        print(f"Error for {date}: {e}")
        return None  # Return None on error

def parse_response(text):
    if text.startswith("Interest rates"):
        scores = [int(str_[0]) for str_ in text.split(": ")[1:]]
    else :
        scores = [int(str_[0]) for str_ in text.replace(" ", "").split(",")]
    return scores

    
def main(api_key_path="key", output_file="output.csv"):
    # Load OpenAI API key
    with open(api_key_path, "r") as f:
        api_key = f.read()
    client = OpenAI(api_key=api_key)
    
    # Load ECB conferences
    conferences = load_ecb_conferences()
    
    # Collect responses in a list
    responses = []
    for idx, row in tqdm(conferences.iterrows()):
        response = chat_completion(client, user=row["text"], date=row["date"], categories=categories)
        responses.append((idx, response))
        
    raw_output_data = [(idx, parse_response(r.choices[0].message.content), conferences["date"][idx]) for (idx, r) in responses]

    # Convert the raw data into a DataFrame
    df = pd.DataFrame(raw_output_data, columns=["Index", "Scores", "Date"])

    # Split the "Scores" list into separate columns
    for idx, category in enumerate(categories):
        df[category] = df["Scores"].apply(lambda x: x[idx])
        
    df = df.drop(columns=["Scores", "Index"])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    
    df.to_csv(output_file, index=False)