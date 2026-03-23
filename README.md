# # CM3070 Final Project - Personalised Educational Content Recommendation System

## Overview
This project develops a personalised educational content recommendation system for learners. The aim is to recommend relevant learning resources based on learner interactions and resource content.

The project compares four recommendation approaches:
- Popularity-based baseline
- Content-based filtering using TF-IDF
- Collaborative filtering
- Hybrid recommender

The system is evaluated using ranking-based metrics to compare recommendation quality.

## Project Structure

```text
cm3070_project/
├── data/
│   ├── resources.csv
│   ├── learners.csv
│   ├── train_interactions.csv
│   └── test_interactions.csv
├── output/
├── compare_models.py
├── collaborative_filtering_baseline.py
├── hybrid_recommender.py
├── tfidf_content_recommender.py
├── requirements.txt
└── README.md
