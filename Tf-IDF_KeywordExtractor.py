import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize


doc = """HSN is looking for an outstanding Full stack python developer, who is highly proficient in creating fast loading websites — either using a standard CMS backend, or a custom CMS. The developer should have worked on other projects and should be able to showcase what they have done.

*
The skills we are looking for are:

Efficient in Python, with knowledge of at least one Python web framework (Django).
Familiarity with some ORM (Object Relational Mapper) libraries.
Good understanding of server-side templating languages {{such as Jinja 2, Mako, etc depending on your technology stack}}.
Knowledge of user authentication and authorization between multiple systems, servers, and environments.
Understanding of fundamental design principles behind a scalable application.
Understanding of the differences between multiple delivery platforms, such as mobile vs desktop, and optimizing output to match the specific platform.
Should have deployed and maintained a django application in production (preferably one that powered a mobile app) which includes knowledge of nginx configuration and application servers like uwsgi, daphne etc.
Able to create database schemes that represent and support business processes.
Should have worked with nosql systems like mongodb and integrated it in a real world application.
Strong unit test and debugging skills.
Proficient in git version control.
Understanding of automation tools like ansible or jenkins is a plus.
Ability to optimise page load speeds and performances is a big plus.
Basic experience with php (wordpress) is a huge plus.
Strong understanding of key disciplines: data structures and algorithms, databases Operating systems, TCP/HTTP stack, software architecture.
Should be able to translate designs, wireframes and mockups into high quality code.
Strong proficiency in JavaScript should be aware of its quirks, and workarounds including DOM manipulation, the JavaScript object model and strong understanding of web fundamentals.
Strong Architecture design skills involving data modeling and low level class design.
Knowledge of RESTful paradigms and experience with consuming APIs in a high quality web application.
Thorough understanding of React.js and its core principles and experience with popular React.js workflows (such as Flux or Redux)
Should be able to build and develop responsive design.
Familiarity with modern front-end build pipelines and tools including common front-end development tools such as Babel, Webpack, NPM, etc.
Version control tools experience, especially git.
Building reusable components and front-end libraries for future use.
Should be able to optimize components for maximum performance across all modern web browsers.
Should have experience in creating SEO friendly websites.
Identify new technologies that improve product development and the user experience
Collaborate with other team members and stakeholders
Experience in working on Linux platform for development and deployment of websites is an advantage
Works well under fast-paced deadlines
"""


pattern = r'''(?x)          
        (?:[A-Z]\.)+        
      | \w+(?:-\w+)*        
      | \$?\d+(?:\.\d+)?%?  
      | \.\.\.              
      | [][.,;"'?():_`-]   
    '''

doc = remove_stopwords(doc.lower())
sentence_tokens = nltk.sent_tokenize(doc)
lemmatizer = WordNetLemmatizer()

new_tokens = []

for doc_sentence in sentence_tokens:
    document_tokens = regexp_tokenize(doc_sentence, pattern)
    document_tokens = [lemmatizer.lemmatize(
        token) for token in document_tokens]
    new_tokens.append(" ".join(document_tokens))
    print(new_tokens)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
X = vectorizer.fit_transform(new_tokens)
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
