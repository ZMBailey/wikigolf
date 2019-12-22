# wikigolf

## What is WikiGolf?

WikiGolf or Wikipedia Golf, is a game where a player is given a starting wikipedia page and a target wikipedia page. Then the player has to try to navigate to the target page using only links from the current wikipedia article, navigating through as few links as possible. This WikiGolf program attempts to have a computer replicate this process using Natural Language Processing.

## How?

WikiGolf is powered using Frequency Distributions and Word2Vec models. At each stage it creates a Word2Vec model based on the current page and the target page, and then examines each link to see how related the title is to the target.
