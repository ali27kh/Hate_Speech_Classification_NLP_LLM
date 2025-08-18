import plotly.graph_objects as go  # For funnel chart and bar charts
import plotly.express as px  # For bar chart and treemap
from plotly.subplots import make_subplots  # Modern replacement for tools.make_subplots
from nltk.corpus import stopwords  # For stopwords
from collections import Counter, defaultdict  # For word counting and frequency dict
import pandas as pd  # For DataFrame operations
from plotly.offline import iplot  # For displaying plots
import matplotlib.pyplot as plt  # For donut plots
from matplotlib.colors import ListedColormap  # For custom colors
from wordcloud import WordCloud  # Importer WordCloud
import matplotlib.pyplot as plt  # Pour l'affichage des graphiques
from textblob import TextBlob  # For sentiment analysis
import plotly.express as px  # For histogram visualizations


class Data_visualisation:
    def __init__(self, df):
        self.df = df  # Store the DataFrame as an instance variable
        self.STOPWORDS = set(stopwords.words('english'))  # Load stopwords once
        # Calculate polarity, review length, and word count
        self.df['polarity'] = self.df['cleaned_comment'].map(lambda text: TextBlob(text).sentiment.polarity)
        self.df['review_len'] = self.df['cleaned_comment'].astype(str).apply(len)
        self.df['word_count'] = self.df['cleaned_comment'].apply(lambda x: len(str(x).split()))

        

    def visualize_class_counts(self):
        temp = self.df.groupby('class').count()['cleaned_comment'].reset_index().sort_values(by='cleaned_comment', ascending=False)
        return temp.style.background_gradient(cmap='Purples')

    def visualize_funnel_chart(self):
        temp = self.df.groupby('class').count()['cleaned_comment'].reset_index().sort_values(by='cleaned_comment', ascending=False)
        fig = go.Figure(go.Funnelarea(
            text=temp['class'],
            values=temp['cleaned_comment'],
            title={"position": "top center", "text": "Funnel-Chart of Hate Speech Distribution"}
        ))
        fig.show()

    def _get_top_words(self, df_subset=None):
        df_to_use = df_subset if df_subset is not None else self.df
        df_to_use['temp_list'] = df_to_use['cleaned_comment'].apply(
            lambda x: [word for word in str(x).split() if word.lower() not in self.STOPWORDS]
        )
        top = Counter([item for sublist in df_to_use['temp_list'] for item in sublist])
        return pd.DataFrame(top.most_common(20), columns=['Common_words', 'count'])

    def visualize_top_words(self):
        temp = self._get_top_words()
        return temp.style.background_gradient(cmap='Blues')

    def visualize_word_bar_chart(self):
        temp = self._get_top_words()
        fig = px.bar(temp, x="count", y="Common_words", title='Common Words in all data', 
                     orientation='h', width=700, height=700, color='Common_words')
        fig.show()

    def visualize_word_treemap(self):
        temp = self._get_top_words()
        fig = px.treemap(temp, path=['Common_words'], values='count', title='Tree of Most Common Words')
        fig.show()

    def visualize_hate_speech_words(self):
        df_class_1 = self.df[self.df['class'] == 1]
        temp = self._get_top_words(df_subset=df_class_1)
        return temp.style.background_gradient(cmap='Blues')

    def visualize_hate_speech_treemap(self):
        df_class_1 = self.df[self.df['class'] == 1]
        temp = self._get_top_words(df_subset=df_class_1)
        fig = px.treemap(temp, path=['Common_words'], values='count', 
                         title='Tree Of Most Common Words for hate speech')
        fig.show()

    def visualize_non_hate_speech_words(self):
        df_class_0 = self.df[self.df['class'] == 0]
        temp = self._get_top_words(df_subset=df_class_0)
        return temp.style.background_gradient(cmap='Blues')

    def visualize_non_hate_speech_treemap(self):
        df_class_0 = self.df[self.df['class'] == 0]
        temp = self._get_top_words(df_subset=df_class_0)
        fig = px.treemap(temp, path=['Common_words'], values='count', 
                         title='Tree Of Most Common Words for non hate speech')
        fig.show()

    def generate_ngrams(self, text, n_gram=1):
        # Custom function for n-gram generation
        token = [token for token in text.lower().split(" ") if token != "" if token not in self.STOPWORDS]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    def horizontal_bar_chart(self, df, color):
        # Custom function for horizontal bar chart
        trace = go.Bar(
            y=df["word"].values[::-1],
            x=df["wordcount"].values[::-1],
            showlegend=False,
            orientation='h',
            marker=dict(color=color),
        )
        return trace

    def visualize_hate_non_hate_bar_charts(self):
        # Filter data for hate (class == 1) and non-hate (class == 0)
        df_pos = self.df[self.df["class"] == 1].dropna()  # Hate speech
        df_neg = self.df[self.df["class"] == 0].dropna()  # Non-hate speech

        # Get word frequencies for hate speech (unigrams)
        freq_dict = defaultdict(int)
        for sent in df_pos["cleaned_comment"]:
            for word in self.generate_ngrams(sent):
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace0 = self.horizontal_bar_chart(fd_sorted.head(25), 'green')

        # Get word frequencies for non-hate speech (unigrams)
        freq_dict = defaultdict(int)
        for sent in df_neg["cleaned_comment"]:
            for word in self.generate_ngrams(sent):
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace1 = self.horizontal_bar_chart(fd_sorted.head(25), 'red')

        # Create subplots
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.04,
                            subplot_titles=["Frequent words in hate speech comment", 
                                          "Frequent words in non-hate speech comment"])
        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 2, 1)

        # Update layout
        fig.update_layout(height=1000, width=900, paper_bgcolor='rgb(233,233,233)', 
                         title="Word Count Plots for Hate Speech and Non-Hate Speech")
        iplot(fig)

    def visualize_hate_non_hate_bigram_bar_charts(self):
        # Filter data for hate (class == 1) and non-hate (class == 0)
        df_pos = self.df[self.df["class"] == 1].dropna()  # Hate speech
        df_neg = self.df[self.df["class"] == 0].dropna()  # Non-hate speech

        # Get bigram frequencies for hate speech
        freq_dict = defaultdict(int)
        for sent in df_pos["cleaned_comment"]:
            for word in self.generate_ngrams(sent, n_gram=2):  # Using bigrams
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace0 = self.horizontal_bar_chart(fd_sorted.head(25), 'green')

        # Get bigram frequencies for non-hate speech
        freq_dict = defaultdict(int)
        for sent in df_neg["cleaned_comment"]:
            for word in self.generate_ngrams(sent, n_gram=2):  # Using bigrams
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace1 = self.horizontal_bar_chart(fd_sorted.head(25), 'brown')

        # Create subplots
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.04, horizontal_spacing=0.25,
                            subplot_titles=["Bigram plots of Hate Speech comment", 
                                          "Bigram plots of Non-Hate Speech comment"])
        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 2, 1)

        # Update layout
        fig.update_layout(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', 
                         title="Bigram Plots for Hate Speech vs Non-Hate Speech")
        iplot(fig)

    def visualize_hate_non_hate_trigram_bar_charts(self):
        # Filter data for hate (class == 1) and non-hate (class == 0)
        df_pos = self.df[self.df["class"] == 1].dropna()  # Positive (Hate speech)
        df_neg = self.df[self.df["class"] == 0].dropna()  # Negative (No hate speech)

        # Get trigram frequencies for hate speech
        freq_dict = defaultdict(int)
        for sent in df_pos["cleaned_comment"]:
            for word in self.generate_ngrams(sent, 3):  # Using trigrams
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace0 = self.horizontal_bar_chart(fd_sorted.head(25), 'green')

        # Get trigram frequencies for non-hate speech
        freq_dict = defaultdict(int)
        for sent in df_neg["cleaned_comment"]:
            for word in self.generate_ngrams(sent, 3):  # Using trigrams
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace1 = self.horizontal_bar_chart(fd_sorted.head(25), 'brown')

        # Create subplots
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.04, horizontal_spacing=0.25,
                            subplot_titles=["Trigram plots of Hate Speech comments", 
                                          "Trigram plots of Non-Hate Speech comments"])
        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 2, 1)

        # Update layout
        fig.update_layout(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Trigram Plots for Hate Speech vs Non-Hate Speech")
        iplot(fig)

    def words_unique(self, sentiment, numwords):
        """
        Identifies words unique to a specific sentiment class in the dataset.

        Parameters:
            sentiment (int): Sentiment class (1 for hate speech, 0 for non-hate speech)
            numwords (int): Number of unique words to display

        Returns:
            DataFrame: Unique words with their frequencies in descending order
        """
        # Get all words from other sentiment classes
        all_other = []
        for item in self.df[self.df['class'] != sentiment]['cleaned_comment']:
            for word in str(item).split():
                all_other.append(word.lower())

        all_other = set(all_other)  # Unique words from other classes

        # Get raw words from the dataset
        raw_text = [word for word_list in self.df['cleaned_comment'] for word in str(word_list).split()]

        # Get words unique to the chosen sentiment class
        specific_only = [word for word in raw_text if word.lower() not in all_other and word.lower() not in self.STOPWORDS]

        mycounter = Counter()

        for item in self.df[self.df['class'] == sentiment]['cleaned_comment']:
            for word in str(item).split():
                if word.lower() in specific_only:
                    mycounter[word.lower()] += 1

        # Convert to DataFrame and return the top words
        unique_words = pd.DataFrame(mycounter.most_common(numwords), columns=['words', 'count'])
        
        return unique_words

    def visualize_unique_words_donut(self, sentiment, numwords, colors):
        """
        Visualizes unique words for a specific sentiment class as a donut plot.

        Parameters:
            sentiment (int): Sentiment class (1 for hate speech, 0 for non-hate speech)
            numwords (int): Number of unique words to display
            colors (list): List of colors for the donut plot
        """
        unique_words = self.words_unique(sentiment, numwords)

        plt.figure(figsize=(16, 10))
        my_circle = plt.Circle((0, 0), 0.7, color='white')
        plt.pie(unique_words['count'], labels=unique_words.words, colors=colors)
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.title(f'DoNut Plot Of Unique {"Hate" if sentiment == 1 else "Non-Hate"} Speech Words')
        plt.show()
    
    def generate_wordcloud_non_hate(self):
        """
        Génère un nuage de mots pour les commentaires non haineux (class == 0).
        """
        text = self.df[self.df["class"] == 0]["cleaned_comment"]  # Filtrer les commentaires non haineux
        wordcloud = WordCloud(
            width=3000,
            height=2000,
            background_color='black',
            stopwords=self.STOPWORDS
        ).generate(" ".join(text.astype(str)))  # Convertir en chaîne et générer le WordCloud

        # Afficher le nuage de mots
        fig = plt.figure(
            figsize=(40, 30),
            facecolor='k',
            edgecolor='k'
        )
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    def generate_wordcloud_hate(self):
        """
        Génère un nuage de mots pour les commentaires haineux (class == 1).
        """
        text = self.df[self.df["class"] == 1]["cleaned_comment"]  # Filtrer les commentaires haineux
        wordcloud = WordCloud(
            width=3000,
            height=2000,
            background_color='black',
            stopwords=self.STOPWORDS
        ).generate(" ".join(text.astype(str)))  # Convertir en chaîne et générer le WordCloud

        # Afficher le nuage de mots
        fig = plt.figure(
            figsize=(40, 30),
            facecolor='k',
            edgecolor='k'
        )
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
    
    def visualize_polarity_distribution(self):
        """
        Visualizes the distribution of sentiment polarity using a histogram.
        """
        fig = px.histogram(
            self.df, 
            x='polarity', 
            nbins=50, 
            title='Sentiment Polarity Distribution',
            labels={'polarity': 'Polarity', 'count': 'Count'},
            color_discrete_sequence=['blue']  # Set bar color to blue
        )
        # Customize the line (border) color of the bars
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig.show()

    def visualize_review_length_distribution(self):
        """
        Visualizes the distribution of review text length using a histogram.
        """
        fig = px.histogram(
            self.df, 
            x='review_len', 
            nbins=100, 
            title='Review Text Length Distribution',
            labels={'review_len': 'Review Length', 'count': 'Count'},
            color_discrete_sequence=['blue']  # Set bar color to blue
        )
        # Customize the line (border) color of the bars
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig.show()

    def visualize_word_count_distribution(self):
        """
        Visualizes the distribution of word count in reviews using a histogram.
        """
        fig = px.histogram(
            self.df, 
            x='word_count', 
            nbins=100, 
            title='Review Text Word Count Distribution',
            labels={'word_count': 'Word Count', 'count': 'Count'},
            color_discrete_sequence=['blue']  # Set bar color to blue
        )
        # Customize the line (border) color of the bars
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig.show()