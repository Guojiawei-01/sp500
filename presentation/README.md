# Presentation Dashboard

This folder contains the static dashboard used as the project presentation aid. It is designed as a non-slide walkthrough of the capstone story: data preparation, sentiment analysis, modeling, backtesting, regime analysis, and final takeaways.

## Files

```text
presentation/
├── index.html
├── styles.css
├── app.js
├── video/
│   └── meeting_04.mp4
└── assets/figures/
```

The dashboard uses static project figures exported from the notebooks and scripts. No server-side code or build step is required.

The recorded group presentation is stored at:

```text
presentation/video/meeting_04.mp4
```

## How to View

Open `presentation/index.html` in a web browser.

If a local server is preferred:

```bash
python -m http.server 8000
```

Then visit:

```text
http://localhost:8000/presentation/
```

## Speaking Flow

The final section includes one suggested 8-12 minute speaking order:

1. Jiawei Guo: problem, dataset, cleaning, limitations
2. Chaoran Chen: sentiment methods and topic features
3. Haoyang Liu: modeling design and test metrics
4. Haonan Li: backtest, regimes, robustness, failure cases
5. Duli Lei: conclusion, next steps, final wrap-up
