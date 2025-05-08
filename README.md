#  Project Proposal

##  Project Title:
**Predicting and Generating Missing Metadata Fields in Streaming Content Using Text-Based Models**

##  Abstract
This project aims to fill missing metadata fields—such as genre, target age group, and description—in streaming content datasets using Natural Language Processing and machine learning techniques. When one of the key elements (genre, age group, or description) is missing, the remaining available fields will be used to predict or generate the missing information. Classification models will be trained for genre and age group prediction, while a generative model will be used to construct meaningful descriptions from title, genre, and age data. Additionally, the relationship between content duration and popularity will be analyzed to understand audience preferences and platform strategies. The project focuses entirely on text-based features and uses publicly available streaming content datasets.

##  Scope of the Project

The project focuses on developing a text-based pipeline that can intelligently complete missing metadata in streaming content datasets. Rather than relying on fixed fields, the system dynamically adapts to the available inputs. It uses classification models to infer categorical fields like genre and audience group, and a generative model to reconstruct missing descriptions. In cases where all three metadata fields are present, the system can also be used to evaluate consistency or enrich existing data for recommendation purposes. This modular structure allows each model to operate independently, ensuring that the pipeline remains useful even when only partial data is available.

##  Research Questions
1. How accurately can the genre of streaming content be predicted from its description and age group?  
2. How accurately can the target age group be predicted from its description and genre?  
3. To what extent can a content description be generated using only the title, genre, and age group?  
4. How does the duration of streaming content (short vs. long) affect popularity among different audience groups, especially younger viewers influenced by short-form media trends?  
5. Are streaming platforms increasingly favoring shorter or more popular content in response to changing viewing habits?

#  About Preprocessing Data

##  Datasets to Be Used

The project utilizes data from four major sources, each serving a distinct purpose in the training pipeline:

| **Dataset**       | **Filename**                 | **Purpose**                                      |
|-------------------|------------------------------|--------------------------------------------------|
| Netflix Titles    | `netflix_titles.csv`         | Source of descriptions, genres, and age ratings |
| Amazon Prime      | `amazon_prime_titles.csv`    | Complementary source for genre and age data     |
| IMDb Movie Data   | `imdb_movie_dataset.csv`     | Additional descriptions and genre diversity     |
| TMDb Metadata     | `tmdb_raw_data.csv`          | Contains popularity metrics, duration, and votes. Fetched using TMDb API from [themoviedb.org](https://www.themoviedb.org/)|

Each dataset is selectively normalized and merged according to its structure. While Netflix and Amazon are used mainly for classification tasks (genre and audience), TMDb is used for analyzing numeric trends like content popularity and duration. IMDb provides additional description and genre information to support the genre classification model.

This section explains how the raw data from multiple streaming content platforms was processed and transformed into machine learning-ready formats for genre classification, audience group prediction, and duration-popularity analysis.

---

## 🔹 Step 1: Define Normalization Maps

A dictionary named `NORMALIZATION_MAPS` was created to ensure consistency across sources:

- **Genre mapping**: Converts vague or platform-specific genres (e.g., "tv dramas") to standardized ones like `"drama"`.
- **Age rating mapping**: Groups ratings (e.g., "TV-14") into `"child"`, `"teen"`, or `"adult"`.
- **Column aliasing**: Supports flexible matching of column names.
- **Allowed genres**: Restricts training data to selected target genres.

---

## 🔹 Step 2: Create Utility Functions

Several utility functions were implemented to standardize the values of key metadata fields across diverse datasets:

- `normalize_title(title)`: Cleans a title string by converting it to lowercase, removing punctuation and articles like "the", "a", or "an", and collapsing whitespace. This ensures that title-based joins between datasets are reliable.
- `normalize_rating_to_group(rating)`: Maps specific age ratings (e.g., "PG-13", "TV-MA") into broader audience categories: "child", "teen", or "adult".
- `normalize_genres_str(genre_string)`: Splits and cleans genre strings, maps ambiguous or compound genres to standardized forms (e.g., "tv comedies" → "comedy").
- `categorize_duration(mins)`: Categorizes numeric durations into labels: "short" (<80 mins), "medium" (80–120 mins), or "long" (>120 mins).

These modular functions were applied later during row-level normalization to ensure all content was consistently and meaningfully represented.

---

## 🔹 Step 3: Normalize Rows from Different Sources

Each source dataset uses different naming conventions and metadata formats. To handle this, we applied row-level normalization functions specific to each type of dataset:

- `normalize_tmdb_row()`: Extracts and transforms TMDb fields such as `vote_average`, `popularity`, and `duration`, while also standardizing genres and title.
- `normalize_genre_row()`: Processes datasets (e.g., Netflix, Amazon, IMDb) by extracting the description and converting genre strings into a cleaned list.
- `normalize_age_row()`: Retrieves content descriptions and maps raw rating labels into unified age categories.

Each row in a dataset is transformed via the appropriate function using `apply()`, creating structured and ready-to-merge records across all sources.

---

## 🔹 Step 4: Load and Merge Datasets

The loading system is designed to handle variations in column naming and structure across different CSV files. It works as follows:

- `load_dataset()` identifies relevant columns using aliases (e.g., "listed_in", "Genre" → "genre") and drops rows with missing critical fields.
- `load_and_merge()` applies the relevant normalization function and concatenates all valid entries into a single DataFrame for each use case.

This step results in three main DataFrames:

- `df_age`: Contains records with description and audience group, used for training age classification models.
- `df_genre`: Contains description and cleaned genre labels, used for genre prediction.
- `df_duration`: Includes numeric fields like duration, votes, and popularity, joined with audience group, for popularity analysis.

---

## 🔹 Step 5: Save Cleaned Datasets

Each processed dataset was saved as a CSV file:

- `audience_group_training_data.csv`
- `genre_training_data.csv`
- `duration_popularity_training_data.csv`

Each contains cleaned, normalized fields such as `description`, `normalized_genres`, `duration`, and `audience_group`.

---


##  Generated Datasets



The preprocessing script produces the following structured CSV files, each tailored for a specific task in the project:

| **File Name**                         | **Content Description**                                                                 | **Used For**                                                              |
|--------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `audience_group_training_data.csv`   | `norm_title`, `description`, `audience_group`                                            | Base dataset for age group classification                                 |
| `genre_training_data.csv`            | `norm_title`, `description`, `normalized_genres`                                         | Base dataset for genre classification                                     |
| `duration_popularity_training_data.csv` | `title`, `norm_title`, `duration`, `release_year`, `vote_average`, `popularity`, `normalized_genres`, `audience_group` | Input for duration and popularity analysis        |
| `audience_group_training_encoded.csv`| One-hot encoded version of `audience_group_training_data.csv`                            | Final input for training age group classification model                   |
| `genre_training_encoded.csv`         | One-hot encoded version of `genre_training_data.csv`                                     | Final input for training genre classification model                       |
| `final_encoded_duration_dataset.csv` | One-hot encoded version of `duration_popularity_training_data.csv`, includes encoded genres and age group | Used in feature-based modeling and popularity-duration exploration |



This improved model accuracy and reduced label noise.

---


## 🔹 Step 6: Genre Filtering (Post-Encoding)

Some genres were removed after encoding due to ambiguity or low semantic value, such as:

- `"tv shows"`, `"movies"`, `"reality tv"`, `"international"`, `"cult"`

These were excluded using `clean_and_filter_genres()`.

Only the following genres were retained:

```
["action", "adventure", "drama", "comedy", "romance", "horror", "documentary", "sci-fi", "thriller"]
```

## 🔹 Step 7: One-Hot Encoding

To prepare data for machine learning models, categorical fields were transformed into binary vectors:

- `normalized_genres` (multi-label) was encoded using `MultiLabelBinarizer`, generating one column per genre.
- `audience_group` and `duration_group` were encoded using `pd.get_dummies()` to convert each category into a binary column.

This encoding ensures compatibility with classification algorithms, which typically require numerical input. The final encoded datasets serve as input for genre and age group prediction, as well as for correlation studies involving duration and popularity.

![Audience Group Distribution](fig_audience_group.png)

![Genre Distribution](fig_genre_distribution.png)

![Duration Group Distribution](fig_duration_distribution.png)

---

##  Final Outcome

The preprocessing pipeline lays the foundation for a modular system designed to **predict or generate missing metadata fields** in streaming content datasets. After cleaning, merging, and encoding:

- The datasets are structured to support three types of tasks:
  1. **Classification Models**:
     - Predict **genre** from `description` and `audience_group`.
     - Predict **audience_group** from `description` and `normalized_genres`.
  2. **Text Generation Model**:
     - Generate **description** from `title`, `normalized_genres`, and `audience_group` using text-based features.
  3. **Analytical Model**:
     - Analyze how **duration** relates to **popularity** across different audience segments.

- Each model is triggered **conditionally**, depending on which metadata field is missing.
- The system supports practical use cases such as:
  - **Metadata completion** for new content
  - **Improved catalog structuring**
  - **Personalized content tagging**

All models rely solely on **textual features**, in line with the project’s NLP focus.


## 🟢 How to Run

To run the project:

1. Make sure you have Python installed (version 3.8+ recommended).
2. Open a terminal or command prompt in the project folder.
3. Install the required libraries by running:

```bash
pip install -r requirements.txt