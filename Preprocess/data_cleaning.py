import re
import pandas as pd
import string

class DataCleaning:
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
        text = DataCleaning.remove_spaces_and_newlines(text)  # Remove spaces, tabs, and newlines
        text = DataCleaning.remove_words_with_numbers(text)  # Remove words with numbers
        text = DataCleaning.remove_punctuation(text)  # Remove punctuation
        text = DataCleaning.remove_unnecessary_spaces(text)  # Remove spaces
        
        return text
