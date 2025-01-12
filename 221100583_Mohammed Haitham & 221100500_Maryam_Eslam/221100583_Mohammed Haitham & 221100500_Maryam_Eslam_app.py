# app.py

import streamlit as st
from recommender import (
    data,
    all_genres_sorted,
    recommend_games
)

def main():
    st.title("Game Recommender System")

    # Age selection
    age_choice = st.radio("Select your age group:", ["above_15", "below_15"])

    # Game options
    game_options = sorted(data["name"].unique().tolist())
    selected_games = st.multiselect(
        "Select exactly 5 games you have played:",
        game_options,
        default=[],
        max_selections=5
    )

    # Genre options
    chosen_genres = st.multiselect(
        "Select up to 3 favorite genres:",
        all_genres_sorted,
        default=[],
        max_selections=3
    )

    # Recommendation button
    if st.button("Get Recommendations"):
        # Validation
        if len(selected_games) < 5:
            st.warning("Please select 5 games.")
        elif len(chosen_genres) < 3:
            st.warning("Please select 3 genres.")
        else:
            # Generate recommendations
            top_recs = recommend_games(
                user_age=age_choice,
                user_games=selected_games,
                user_genres=chosen_genres,
                top_n=5
            )

            # Display recommendations
            st.subheader("Recommended Games:")
            for idx, row in top_recs.iterrows():
                st.write("--------------------------------------------------")
                st.write(f"**Name**: {row['unified_name']}")
                st.write(f"**Required Age**: {row['required_age']}")
                st.write(f"**Rating**: {row['ratings']:.1f}")
                st.write(f"**Is Free**: {row['is_free']}")
                st.write(f"**Supported Languages**: {row['supported_languages']}")
                st.write(f"**Website**: {row['website']}")
                st.write(f"**Categories**: {row['categories']}")

                # Show image if available
                if isinstance(row["header_image"], str) and row["header_image"].strip():
                    st.image(row["header_image"], caption=row["unified_name"])

                st.write(f"**Similarity Score**: {row['similarity_score']:.3f}")
                st.write(f"**Genre Overlap**: {row['genre_overlap']:.3f}")
                st.write(f"**Combined Score**: {row['combined_score']:.3f}")
                st.write("**Explanation**:")

                # Display explanation line by line
                explanation_lines = row["similarity_details"].split("\n")
                for expl_line in explanation_lines:
                    st.write(expl_line)

if __name__ == "__main__":
    main()
