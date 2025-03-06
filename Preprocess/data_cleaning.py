import re
import pandas as pd
import string
import nltk
from nltk.corpus import words
import language_tool_python

# Download the words corpus if not already downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

tool = language_tool_python.LanguageTool('en-US', remote_server='https://api.languagetool.org')
class DataCleaning:
    # Dictionary of common abbreviations
    ABBREVIATIONS = {
        "dont": "do not",
        "cant": "cannot",
        "wont": "will not",
        "ive": "i have",
        "im": "i am",
        "id": "i would",
        "ill": "i will",
        "yall": "you all",
        "its": "it is",
        "hes": "he is",
        "shes": "she is",
        "theyre": "they are",
        "whats": "what is",
        "wheres": "where is",
        "theres": "there is",
        "didnt": "did not",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "mr": "mister",
        "mrs": "missus",
        "ms": "miss",
        "dr": "doctor",
        "prof": "professor",
        "approx": "approximately",
        "etc": "et cetera",
        "vs": "versus",
        "dept": "department",
        "govt": "government",
        "info": "information",
        "no": "number",
        "tel": "telephone",
        "u": "you",
        "thats": "that's",
        "ur": "you're"
    }


    @staticmethod
    def clean_class_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Cleans the class column by assigning:
        - 1 if value is between 0.6 and 1
        - 0 if value is between 0 and 0.4
        - not considered if value is between 0.39 and 0.59 
        """
        df['class'] = df[column_name].apply(lambda x: 1 if 0.6 <= x <= 1 else (0 if 0 <= x <= 0.4 else 'not considered'))
        return df
    
    @staticmethod
    def remove_links(text):
        # Regular expression to find and remove links
        link_pattern = re.compile(r'http\S+|www\S+')
        return link_pattern.sub(r'', text)

    @staticmethod
    def remove_special_chars(text):
        # Regular expression to remove special characters
        special_chars_pattern = re.compile(r'[^\w\s]')
        return re.sub(special_chars_pattern, '', text)

    @staticmethod
    def remove_emojis(text):
        # Regular expression to remove emojis
        emoji_pattern = re.compile(
            u"[" 
            u"\U0001F600-\U0001F64F"  # emoticons 
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs 
            u"\U0001F680-\U0001F6FF"  # transport & map symbols 
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS) 
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_square_brackets(text):
        # Regular expression to remove text inside square brackets
        return re.sub(r'\[.*?\]', '', text)

    @staticmethod
    def remove_hashtags(text):
        # Regular expression to remove hashtags
        return re.sub(r'#\w+', '', text)

    @staticmethod
    def remove_spaces_and_newlines(text):
        # Replace spaces, tabs, and newlines by a string empty
        return text.replace("\t", "").replace("\n", "")

    @staticmethod
    def remove_words_with_numbers(text):
        # Regular expression to remove words with numbers
        return re.sub(r'\w*\d\w*', '', text)
    
    @staticmethod
    def remove_html_tags(text):
        # Regular expression to remove HTML tags
        return re.sub(r'<.*?>+', '', text)

    @staticmethod
    def remove_html_entities(text):
        # Regular expression to remove HTML entities
        text = re.sub(r'&gt;', '', text)  # greater than sign
        text = re.sub(r'&#x27;', "'", text)  # apostrophe
        text = re.sub(r'&#x2F;', ' ', text)
        text = re.sub(r'&#62;', '', text)
        return text
    
    @staticmethod
    def remove_specific_tags(text):
        # Regular expression to remove specific tags like <a>, <p>, <i>
        text = re.sub(r'<p>', ' ', text)  # paragraph tag
        text = re.sub(r'<i>', ' ', text)  # italics tag
        text = re.sub(r'</i>', '', text)
        text = re.sub(r'<a[^>]*>(.+?)</a>', 'Link.', text)
        return text

    @staticmethod
    def remove_punctuation(text):
        # Remove punctuation using string.punctuation
        return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    @staticmethod
    def remove_unnecessary_spaces(text):
        # Remove leading and trailing spaces, and replace multiple consecutive spaces with a single space
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_user_mentions(text):
        # Regular expression to remove @user mentions
        return re.sub(r'@\w+', '', text)
    
    @staticmethod
    def expand_abbreviations(text):
        """
        Expands common abbreviations in the text to their full forms.
        
        Args:
            text (str): Input text containing abbreviations
            
        Returns:
            str: Text with abbreviations expanded
        """
        # Split text into words
        words = text.split()
        
        # Replace abbreviations with their full forms
        expanded_words = [DataCleaning.ABBREVIATIONS.get(word.lower(), word) for word in words]
        
        # Join words back together
        return ' '.join(expanded_words)
    
    @staticmethod
    def display_oov_words(text):
        """
        Detects out-of-vocabulary words in a given text using NLTK's word corpus.
        
        Args:
        text (str): Input text to analyze
            
        Returns:
        dict: Contains original text, OOV words found, and vocabulary stats
        """
        # Convert NLTK words to a set for faster lookup
        vocabulary = set(words.words())
            
        # Convert text to lowercase and split into words
        # Remove punctuation and split by whitespace
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        input_words = set(cleaned_text.split())
            
        # Find words not in vocabulary
        oov_words = [word for word in input_words if word not in vocabulary]
            
        # Prepare results
        result = {
            'oov_words': oov_words,
            'oov_count': len(oov_words),
        }
            
        return result
    
    @staticmethod
    def correct_grammar(text):
        """
        Corrects grammatical errors in the text using language_tool_python.
        
        Args:
            text (str): Input text to correct
            
        Returns:
            str: Grammatically corrected text
        """
        try:
            # Use the language tool to correct the text
            corrected_text = tool.correct(text)
            return corrected_text
        except Exception as e:
            print(f"Error in grammar correction: {e}")
            return text  # Return original text if correction fails
    
    
    @staticmethod
    def detect_oov_words(text):
        """
        Removes out-of-vocabulary words from the given text using NLTK's word corpus.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Text with OOV words removed
        """
        # Convert NLTK words to a set for faster lookup
        vocabulary = set(words.words())
        
        # Convert text to lowercase and split into words
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        input_words = cleaned_text.split()
        
        # Keep only words that are in vocabulary
        cleaned_words = [word for word in input_words if word in vocabulary]
        
        # Join the words back together with spaces
        return ' '.join(cleaned_words)
    


    @staticmethod
    def preprocess_text(text):
        """
        Preprocesses the text by applying all the necessary text cleaning operations.
        """
        text = str(text).lower()  # Convert text to lowercase
        text = DataCleaning.remove_links(text)  # Remove links
        text = DataCleaning.remove_user_mentions(text)  # Remove @user mentions
        text = DataCleaning.remove_html_tags(text)  # Remove HTML tags
        text = DataCleaning.remove_html_entities(text)  # Remove HTML entities
        text = DataCleaning.remove_specific_tags(text)  # Remove specific tags
        text = DataCleaning.remove_special_chars(text)  # Remove special characters
        text = DataCleaning.remove_emojis(text)  # Remove emojis
        text = DataCleaning.remove_square_brackets(text)  # Remove text inside square brackets
        text = DataCleaning.remove_hashtags(text)  # Remove hashtags
        text = DataCleaning.expand_abbreviations(text)  # Correct abbreviation 
        text = DataCleaning.remove_spaces_and_newlines(text)  # Remove spaces, tabs, and newlines
        text = DataCleaning.remove_words_with_numbers(text)  # Remove words with numbers
        text = DataCleaning.remove_punctuation(text)  # Remove punctuation
        # text = DataCleaning.detect_oov_words(text)  # Remove OOV words (updated function)
        text = DataCleaning.remove_unnecessary_spaces(text)  # Remove spaces
        # text = DataCleaning.correct_grammar(text)  # correct grammar


        
        
        return text
