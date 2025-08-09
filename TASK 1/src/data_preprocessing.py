"""
Data Preprocessing Module for Movie Genre Classification
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_real_dataset(self):
        """
        Load the real movie dataset from the Genre Classification Dataset folder
        """
        try:
            # Load training data
            train_file = 'Genre Classification Dataset/train_data.txt'
            test_file = 'Genre Classification Dataset/test_data.txt'
            
            # Read training data
            train_data = []
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and ':::' in line:
                        parts = line.split(':::')
                        if len(parts) >= 4:
                            movie_id = parts[0].strip()
                            title = parts[1].strip()
                            genre = parts[2].strip()
                            description = parts[3].strip()
                            train_data.append({
                                'id': movie_id,
                                'title': title,
                                'genre': genre,
                                'plot': description
                            })
            
            # Read test data (without genres)
            test_data = []
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and ':::' in line:
                        parts = line.split(':::')
                        if len(parts) >= 3:
                            movie_id = parts[0].strip()
                            title = parts[1].strip()
                            description = parts[2].strip()
                            test_data.append({
                                'id': movie_id,
                                'title': title,
                                'plot': description
                            })
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            
            print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
            print(f"Training genres: {train_df['genre'].nunique()} unique genres")
            print(f"Genre distribution in training data:")
            print(train_df['genre'].value_counts().head(10))
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error loading real dataset: {e}")
            print("Falling back to sample data...")
            return self.load_sample_data(), None
    
    def load_sample_data(self):
        """
        Create sample movie data if no dataset is available
        """
        sample_data = {
            'title': [
                'The Dark Knight', 'Titanic', 'The Shawshank Redemption',
                'Inception', 'The Lion King', 'Jurassic Park', 'Forrest Gump',
                'The Matrix', 'Pulp Fiction', 'The Godfather', 'Avatar',
                'The Avengers', 'Iron Man', 'Spider-Man', 'Batman Begins',
                'The Notebook', 'La La Land', 'Casablanca', 'Gone with the Wind',
                'The Wizard of Oz', 'Jaws', 'The Exorcist', 'Halloween',
                'A Nightmare on Elm Street', 'The Silence of the Lambs',
                'Toy Story', 'Finding Nemo', 'Monsters Inc', 'Up', 'Frozen',
                'The Hangover', 'Bridesmaids', 'Superbad', 'Anchorman',
                'The Grand Budapest Hotel', 'The Big Lebowski', 'Fargo'
            ],
            'plot': [
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
                'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
                'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
                'A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the park\'s cloned dinosaurs to run loose.',
                'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.',
                'A computer programmer discovers that reality as he knows it is a simulation created by machines, and joins a rebellion to break free.',
                'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
                'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.',
                'Earth\'s mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity.',
                'After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor to fight evil.',
                'After being bitten by a genetically-modified spider, a shy teenager gains spider-like abilities that he eventually must use to fight evil as a superhero.',
                'After training with his mentor, Batman begins his fight to free crime-ridden Gotham City from corruption.',
                'A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom, but they are soon separated because of their social differences.',
                'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.',
                'A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband escape the Nazis in Morocco.',
                'A manipulative woman and a roguish man conduct a turbulent romance during the American Civil War and Reconstruction periods.',
                'Dorothy Gale is swept away from a farm in Kansas to a magical land of Oz in a tornado and embarks on a quest with her new friends to see the Wizard.',
                'When a killer shark unleashes chaos on a beach community, it\'s up to a local sheriff, a marine biologist, and an old seafarer to hunt the beast down.',
                'When a 12-year-old girl is possessed by a mysterious entity, her mother seeks the help of two priests to save her.',
                'Fifteen years after murdering his sister on Halloween night 1963, Michael Myers escapes from a mental hospital and returns to the small town of Haddonfield to kill again.',
                'The monstrous spirit of a slain child murderer seeks revenge by invading the dreams of teenagers whose parents were responsible for his untimely death.',
                'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.',
                'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
                'After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.',
                'In order to power the city, monsters have to scare children so that they scream. However, the children are toxic to the monsters, so after the children are scared, the monsters have to let them go.',
                'Seventy-eight year old Carl Fredricksen travels to Paradise Falls in his home equipped with balloons, inadvertently taking a young stowaway.',
                'When the newly crowned Queen Elsa accidentally uses her power to turn things into ice to curse her home in infinite winter, her sister Anna teams up with a mountain man, his playful reindeer, and a snowman to change the weather condition.',
                'Three buddies wake up from a bachelor party in Las Vegas, with no memory of the previous night and the bachelor missing. They make their way around the city in order to find their friend before his wedding.',
                'Competition between the maid of honor and a bridesmaid, over who is the bride\'s best friend, threatens to upend the life of an out-of-work pastry chef.',
                'Three friends try to get alcohol for a party, but their plan goes awry when they are caught by the police.',
                'In the 1970s, an anchorman\'s stint as San Diego\'s top-rated newsreader is challenged when an ambitious woman is hired as his co-anchor.',
                'A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel\'s glorious years under an exceptional concierge.',
                'Jeff The Dude Lebowski, a Los Angeles slacker who only wants to bowl and drink white Russians, is mistaken for another Jeffrey Lebowski, a wheelchair-bound millionaire, and finds himself dragged into a strange series of events.',
                'Jerry Lundegaard\'s inept crime falls apart due to his and his henchmen\'s bungling and the persistent police work of the quite pregnant Marge Gunderson.'
            ],
            'genre': [
                'Action', 'Romance', 'Drama', 'Sci-Fi', 'Animation', 'Adventure',
                'Drama', 'Sci-Fi', 'Crime', 'Crime', 'Sci-Fi', 'Action',
                'Action', 'Action', 'Action', 'Romance', 'Romance', 'Romance',
                'Romance', 'Adventure', 'Horror', 'Horror', 'Horror',
                'Horror', 'Thriller', 'Animation', 'Animation', 'Animation',
                'Animation', 'Animation', 'Comedy', 'Comedy', 'Comedy',
                'Comedy', 'Comedy', 'Comedy', 'Comedy'
            ]
        }
        
        return pd.DataFrame(sample_data)
    
    def prepare_data(self, data_path='data/movies.csv'):
        """
        Load, preprocess, and prepare data for training
        """
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Try to load the real dataset first, then fallback to sample data
        try:
            print("Attempting to load real dataset...")
            train_df, test_df = self.load_real_dataset()
            
            if train_df is not None:
                df = train_df
                print("Using real dataset for training")
            else:
                print("Real dataset not available. Using sample data...")
                df = self.load_sample_data()
                df.to_csv(data_path, index=False)
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample data...")
            df = self.load_sample_data()
            df.to_csv(data_path, index=False)
        
        # Clean the plot text
        print("Cleaning text data...")
        df['clean_plot'] = df['plot'].apply(self.clean_text)
        
        # Remove rows with empty plots
        df = df[df['clean_plot'].str.len() > 0]
        
        # Encode genres
        y = self.label_encoder.fit_transform(df['genre'])
        
        # Vectorize the text
        print("Vectorizing text data...")
        X = self.vectorizer.fit_transform(df['clean_plot'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save the preprocessor components
        self.save_preprocessor()
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self):
        """
        Save the preprocessor components for later use
        """
        os.makedirs('models', exist_ok=True)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_preprocessor(self):
        """
        Load the saved preprocessor components
        """
        try:
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            return True
        except FileNotFoundError:
            print("Preprocessor files not found. Please train the model first.")
            return False 