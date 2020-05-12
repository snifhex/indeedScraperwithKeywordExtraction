
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from gensim.parsing.preprocessing import remove_stopwords

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
Works well under fast-paced deadlines"""


doc = remove_stopwords(doc.lower())
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


"""
#KeywordOutput for above document.
{'mobile', 'web', 'templating', 'automation', 'structure', 'object', 'identify', 'consuming', 'mockups', 'server-side', 'tcp', 'npm', 'model', 
'authentication', 'nginx', 'python', 'responsive', 'integrated', 'php', 'building', 'architecture', 'backend', 'authorization', 'core', 'multiple', 'j', 'member', 'jenkins', 'modeling', 
'wordpress', 'debugging', 'orm', 'output', 'product', 'build', 'database', 'stack', 'configuration', 'cm', 'relational', 'uwsgi', 'performance', 'scheme', 'dom', 
'webpack', 'babel', 'class', 'translate', 'scalable', 'environment', 'desktop', 'production', 'seo', 'workflow', 'ansible', 'project', 'nosql', 'daphne', 'process',
 'loading', 'redux', 'http', 'mapper', 'load', 'algorithm', 'pipeline', 'server', 
'deployment', 'key', 'mako', 'browser', 'javascript', 'django', 'flux', 'mongodb', 'basic', 'git', 'platform', 'match', 'future', 'version', 'linux', 'system', 'difference'}
"""
