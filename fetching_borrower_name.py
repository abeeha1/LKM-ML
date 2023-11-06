import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from time import sleep

# Load the CSV file into a DataFrame
df = pd.read_csv("23oct.csv")
url_column = df["SEC.gov URL"]

options = Options()
options.headless = True

driver = webdriver.Firefox(options=options)

model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def extract_borrowers_with_selenium_and_bert(url):
    try:
        # Open the URL in Firefox
        driver.get(url)
        sleep(5)
        page_text = driver.find_element(By.TAG_NAME, 'body').text

        # Extract the first 10 lines from the page and remove white spaces
        lines_10 = [line.strip() for line in page_text.splitlines()[:10]]
        answer_text = ' '.join(lines_10)

        question = "Who are the borrowers?"
        input_ids = tokenizer.encode(question, answer_text)
        attention_mask = [1] * len(input_ids)
        output = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

        start_indexes = torch.argsort(output[0][0, :len(input_ids) - input_ids.index(tokenizer.sep_token_id)], descending=True)
        end_indexes = torch.argsort(output[1][0, :len(input_ids) - input_ids.index(tokenizer.sep_token_id)], descending=True)

        # Initialize lists to store the multiple answers
        answers = []

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index <= end_index:
                    answer = tokenizer.decode(input_ids[start_index:end_index + 1], skip_special_tokens=True)
                    answers.append(answer)

        return answers
    except Exception as e:
        return None

data_from_urls = [extract_borrowers_with_selenium_and_bert(url) for url in url_column]

result_df = pd.DataFrame({"SEC.gov URL": url_column, "Borrower Names": data_from_urls})

result_df.to_csv("result.csv", index=False)

driver.quit()