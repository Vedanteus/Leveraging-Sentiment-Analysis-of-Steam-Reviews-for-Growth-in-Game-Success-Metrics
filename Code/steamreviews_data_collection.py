import steamreviews
import pandas as pd

# List of Steam App IDs and corresponding game names
games = {
    578080: "PUBG: BATTLEGROUNDS",
    271590: "Grand Theft Auto V",
    1245620: "Cyberpunk 2077",
    550: "Left 4 Dead 2",
    730: "Counter-Strike 2",
    105600: "Terraria",
    1091500: "Cyberpunk 2077",
    218620: "PAYDAY 2",
    322330: "Don't Starve Together",
    292030: "The Witcher 3: Wild Hunt"
}

# Initialize the request parameters
request_params = {
    'filter': 'all',         # Fetch all reviews, sorted by helpfulness
    'num_per_page': 100,     # Number of reviews per request (max allowed is 100)
    'language': 'english'    # Only get reviews in English
}

# Maximum number of reviews to collect per app ID
max_reviews = 20000

# List to hold all reviews across different app IDs
all_reviews = []

# Loop through each app ID and collect reviews
for app_id, game_name in games.items():
    reviews_collected = []
    while len(reviews_collected) < max_reviews:
        # Download reviews in chunks
        review_dict, _ = steamreviews.download_reviews_for_app_id(app_id, chosen_request_params=request_params)
        
        # Extract reviews from the current batch
        reviews = review_dict['reviews'].values()
        reviews_collected.extend(reviews)
        
        # Break if fewer reviews were returned than requested (end of available reviews)
        if len(reviews) < request_params['num_per_page']:
            break
    
    # Add app_id and game_name to each review
    for review in reviews_collected[:max_reviews]:
        review_data = {
            'app_id': app_id,
            'game_name': game_name,
            'review': review['review']
        }
        all_reviews.append(review_data)

# Convert the collected reviews to a pandas DataFrame
df = pd.DataFrame(all_reviews)

# Save the DataFrame to a single CSV file
csv_file_path = "steam_reviews_all_games.csv"
df.to_csv(csv_file_path, index=False)

print(f"All reviews saved to {csv_file_path}")