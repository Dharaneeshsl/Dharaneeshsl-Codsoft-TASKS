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
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = word_tokenize(text)
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_real_dataset(self):
        try:
            train_file = 'Genre Classification Dataset/train_data.txt'
            test_file = 'Genre Classification Dataset/test_data.txt'
            
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
            
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            
            print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
            
            return train_df, test_df
        
        except FileNotFoundError:
            print("Real dataset not found in 'Genre Classification Dataset/'.")
            return None, None
    
    def load_sample_data(self):
        sample_data = {
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction',
                'Forrest Gump', 'Inception', 'The Matrix', 'Goodfellas',
                'Star Wars: Episode V - The Empire Strikes Back', 'The Lord of the Rings: The Return of the King',
                'Fight Club', 'The Lord of the Rings: The Fellowship of the Ring', 'The Lord of the Rings: The Two Towers',
                'Interstellar', 'Gladiator', 'Saving Private Ryan', 'The Green Mile',
                'The Silence of the Lambs', 'Se7en', 'City of God', 'Life Is Beautiful',
                'It\'s a Wonderful Life', 'Spirited Away', 'The Lion King', 'Modern Times',
                'Back to the Future', 'Whiplash', 'The Prestige', 'The Departed',
                'The Intouchables', 'Grave of the Fireflies', 'Casablanca', 'Cinema Paradiso',
                'Rear Window', 'Alien', 'Apocalypse Now', 'Memento', 'Raiders of the Lost Ark',
                'Django Unchained', 'WALL·E', 'The Shining', 'Paths of Glory',
                'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb', 'The Great Dictator',
                'Sunset Boulevard', 'The Grand Budapest Hotel', 'Fargo'
            ],
            'plot': [
                'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
                'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
                'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.',
                'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
                'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito in the Italian-American crime syndicate.',
                'After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training with Yoda, while his friends are pursued by Darth Vader.',
                'Gandalf and Aragorn lead the World of Men against Sauron\'s army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.',
                'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.',
                'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.',
                'While Frodo and Sam edge closer to Mordor with the help of the shifty Gollum, the divided fellowship makes a stand against Sauron\'s new ally, Saruman, and his hordes of Isengard.',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.',
                'Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action.',
                'The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder and rape, yet who has a mysterious gift.',
                'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.',
                'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.',
                'In the slums of Rio, two kids\' paths diverge as one struggles to become a photographer and the other a kingpin.',
                'When an open-minded Jewish librarian and his son become victims of the Holocaust, he uses a perfect mixture of will, humor, and imagination to protect his son from the dangers around their camp.',
                'An angel is sent from Heaven to help a desperately frustrated businessman by showing him what life would have been like if he had never existed.',
                'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts.',
                'A Lion cub crown prince is tricked by a treacherous uncle into thinking he caused his father\'s death and flees into exile in despair, only to learn in adulthood his identity and his responsibilities.',
                'The Tramp struggles to live in modern industrial society with the help of a young homeless woman.',
                'Marty McFly, a 17-year-old high school student, is accidentally sent thirty years into the past in a time-traveling DeLorean invented by his close friend, the eccentric scientist Doc Brown.',
                'A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student\'s potential.',
                'After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.',
                'An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.',
                'A quadriplegic aristocrat, who was injured in a paragliding accident, hires a young man from the projects to be his caregiver.',
                'A young boy and his little sister struggle to survive in Japan during World War II.',
                'A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband escape the Nazis in French Morocco.',
                'A filmmaker recalls his childhood when he fell in love with the movies at his village\'s theater and formed a deep friendship with the theater\'s projectionist.',
                'A wheelchair-bound photographer spies on his neighbors from his apartment window and becomes convinced one of them has committed murder.',
                'After a space merchant vessel receives an unknown transmission as a distress call, one of the crew is attacked by a mysterious life form and they soon realize that its life cycle has merely begun.',
                'During the Vietnam War, Captain Willard is sent on a dangerous mission into Cambodia to assassinate a renegade Colonel who has set himself up as a god among a local tribe.',
                'A man with short-term memory loss attempts to track down his wife\'s murderer.',
                'In 1936, archaeologist and adventurer Indiana Jones is hired by the U.S. government to find the Ark of the Covenant before Adolf Hitler\'s Nazis can obtain its awesome powers.',
                'With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner.',
                'In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind.',
                'A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings from both past and future.',
                'After refusing to attack an enemy position, a general accuses the soldiers of cowardice and their commanding officer must defend them.',
                'An insane general triggers a path to nuclear holocaust that a War Room full of politicians and generals frantically tries to stop.',
                'A German-Jewish barber, a victim of Nazi persecution, is mistaken for a dictator and is asked to address the nation.',
                'A screenwriter develops a dangerous relationship with a faded film star determined to make a triumphant return.',
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
        os.makedirs('data', exist_ok=True)
        
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
        
        print("Cleaning text data...")
        df['clean_plot'] = df['plot'].apply(self.clean_text)
        
        df = df[df['clean_plot'].str.len() > 0]
        
        y = self.label_encoder.fit_transform(df['genre'])
        
        print("Vectorizing text data...")
        X = self.vectorizer.fit_transform(df['clean_plot'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.save_preprocessor()
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self):
        os.makedirs('models', exist_ok=True)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_preprocessor(self):
        try:
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            return True
        except FileNotFoundError:
            print("Preprocessor files not found. Please train the model first.")
            return False