"""
Retrieve job listings according to the job title and location from indeed.com

Usage:
    python3 indeedScraper.py <job> <location>

    note: If command line argument not given, script will use defautl for job = "Product+Manager"
        and loc = "San+Francisco%2C+CA".

Requirements:
    modules: os, bs4, csv, pd, nltk, sys, rake, yake, gensim, locale, requests.

"""
import os
import sys
import csv
import yake
import locale
import requests
import pandas as pd
from rake_nltk import Rake
from bs4 import BeautifulSoup as bs
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from gensim.parsing.preprocessing import remove_stopwords


def getTotalResults(job, loc):
    """
    Retrieve total results of job listing from the site.

    Args:
        job: Job title to be searched.
        loc: Job location to be searched.

    Returns:
        A integer which is total number of result.
    """

    resp = requests.get(
        f"https://www.indeed.com/jobs?q={job}&l={loc}")

    soup = bs(resp.text, "lxml")
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    total_results = soup.find(
        "div", id="searchCountPages").text.strip().split()[-2]
    total_results = locale.atoi(total_results)
    print(f"TotalResults: {total_results}")
    return total_results


def scraper(total_results, job, loc):
    """
    Gets all the job data from the site and saves in the "indeedScraping.csv".

    Args:
        total_results: Total number of job listing on the site which can be retrieved by the getTotalResult function.
        job: Job title to be searched.
        loc: Job location to be searched.

    Returns:
        Doesn't returns any value.
    """

    for offset in range(0, total_results, 10):
        url = f"https://www.indeed.com/jobs?q={job}&l={loc}&start={offset}"
        resp = requests.get(url)

        if resp.ok:
            soup = bs(resp.text, "lxml")
            job_listings = soup.find("td", id="resultsCol").find_all(
                "div", class_="result")

            # print(len(job_listings))

            for job_content in job_listings:
                title = job_content.h2.a.get('title')
                link = job_content.h2.a.get("href")
                link = "https://indeed.com"+link
                company = job_content.find(
                    "span", class_="company").text.strip()
                job_location = job_content.find(
                    "div", class_="recJobLoc").get("data-rc-loc")

                resp2 = requests.get(link)

                if resp2.ok:
                    soup2 = bs(resp2.text, "lxml")
                    try:
                        experience = soup2.find_all(
                            "span", class_="jobsearch-JobMetadataHeader-iconLabel")[-1].text.strip()
                        if "experience" not in experience:
                            experience = ""
                    except:
                        experience = ''
                    try:
                        skills = [skill.text.strip() for skill in soup2.find_all(
                            "span", class_="jobsearch-JobMetadataHeader-skillItem")]
                    except:
                        skills = ''

                    description = soup2.find(
                        "div", id="jobDescriptionText").text.strip()

                    doc = remove_stopwords(description.lower())

                    description_keywords_yake = yake_keyword(doc)
                    description_keywords_rake = rake_keyword(doc)
                    description_keywords_soTags = soTags_keyword(doc)

                writeToCsv({"title": title, "company": company, 'job_location': job_location,
                            'experience': experience, 'skills': skills, 'link': link, 'keywords (Yake)': description_keywords_yake, 'keywords (Rake)': description_keywords_rake, 'keywords ( StackOverflow Tags)': description_keywords_soTags})
                print(
                    f"\n{title}\n{company}\n{job_location}\n{experience}\n{skills}\n")


def rake_keyword(doc):
    """
    Extracts keywords from the given text using rake.

    Args:
        doc: Paragraph from keywords need to be extracted.

    Returns:
        Returns Keywords extracted from the text document passed.
    """

    r = Rake()
    r.extract_keywords_from_text(doc)
    keywords = r.get_ranked_phrases()

    return keywords


def yake_keyword(doc):
    """
    Extracts keywords from the given text using yake.

    Args:
        doc: Paragraph from keywords need to be extracted.

    Returns:
        Returns Keywords extracted from the text document passed.
    """

    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 20

    extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                      dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = extractor.extract_keywords(doc)
    keywords = [word for number, word in keywords]
    return keywords


def soTags_keyword(doc):
    """
    Extracts keywords from the given text using yake.

    Args:
        doc: Paragraph from keywords need to be extracted.

    Returns:
        Returns Keywords extracted from the text document passed.
    """
    tags = pd.read_csv('tags.csv')

    pattern = r'''(?x)          
            (?:[A-Z]\.)+        
        | \w+(?:-\w+)*        
        | \$?\d+(?:\.\d+)?%?  
        | \.\.\.              
        | [][.,;"'?():_`-]   
        '''
    document_tokens = regexp_tokenize(doc, pattern)
    lemmatizer = WordNetLemmatizer()

    document_tokens = [lemmatizer.lemmatize(
        token) for token in document_tokens]

    keywords = set(document_tokens) & set(tags.technology)

    return keywords


def writeToCsv(contents):
    """
    Writes one row at a time in csv file.

    Args:
        contents: Dictionary of one row.

    Returns:
        Doesn't returns anything.
    """
    with open("indeedScraping.csv", 'a') as csv_writer:
        fields = ['title', 'company', 'job_location',
                  'experience', 'skills', 'link', 'keywords (Yake)', 'keywords (Rake)', 'keywords ( StackOverflow Tags)']
        writer = csv.DictWriter(csv_writer, fieldnames=fields)
        writer.writerow(contents)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        job = sys.argv[1]
        loc = sys.argv[2]
    elif len(sys.argv) > 1:
        job = sys.argv[1]
        loc = ""
    else:
        job = ""  # Update here new job names.
        loc = ""  # Update here for new locations.

    if not os.path.exists("indeedScraping.csv"):
        with open("indeedScraping.csv", 'w') as csv_writer:
            fields = ['title', 'company', 'job_location',
                      'experience', 'skills', 'link', 'keywords (Yake)', 'keywords (Rake)', 'keywords ( StackOverflow Tags)']
            writer = csv.DictWriter(csv_writer, fieldnames=fields)
            writer.writeheader()

    total_results = getTotalResults(job, loc)
    scraper(total_results, job, loc)
