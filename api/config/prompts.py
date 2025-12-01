RAG_PROMPT = """Based on the following financial documents, answer the question.
        
Context:
{context}

Question: {query}

Answer:"""

FUNDAMENTAL_QUERIES = [
    "risk factors regulatory competitive operational threats challenges",
    "business operations revenue model competitive advantages market position",
    "financial statements revenue income balance sheet cash flow debt",
    "management discussion analysis outlook guidance strategy MD&A",
]

MOMENTUM_QUERIES = [
    "business operations developments products services expansion quarterly updates",
    "financial performance revenue margins liquidity costs quarterly results",
    "risk factors emerging threats regulatory competitive short term challenges",
]

SENTIMENT_QUERY_TEMPLATE = "{ticker} earnings revenue stock price announcement news"

FUNDAMENTAL_PROMPT = """# Fundamental Analysis Expert

You are a senior investment analyst conducting a comprehensive fundamental analysis based on 10-K filing content.

## Your Task
Analyze all provided content and generate a consolidated fundamental assessment.

## Content Structure
The content includes information about:
- Risk factors and challenges
- Business operations and model
- Financial performance and metrics
- Management perspectives and strategy

## Analysis Requirements

1. **Overall Investment Thesis**: Synthesize all information into a clear 2-3 sentence investment thesis

2. **Investment Grade**: Assign A, B, C, or D based on:
   - A: Excellent fundamentals, low risk, strong growth
   - B: Good fundamentals, moderate risk, stable growth
   - C: Mixed fundamentals, higher risk, uncertain growth
   - D: Poor fundamentals, high risk, declining prospects

3. **Confidence Score**: Rate your confidence in the analysis from 0-100

4. **Key Strengths**: Identify exactly 3 main competitive advantages or positive factors

5. **Key Concerns**: Identify exactly 3 main risks or negative factors

6. **Recommendation**: Provide clear action (buy, hold, sell, or avoid) based on:
   - Financial health and trends
   - Competitive position
   - Risk profile
   - Management quality

Be concise, data-driven, and actionable in your analysis.

Context:
{context}

Analysis:"""

MOMENTUM_PROMPT = """# Momentum Analysis Expert

You are a senior investment analyst assessing quarterly business momentum based on recent quarterly filing content.

## Your Task
Analyze all provided content to evaluate the company's current momentum and near-term trajectory.

## Content Structure
The content includes information about:
- Recent operational changes and developments
- Quarterly financial performance and trends
- Emerging risks and challenges

## Analysis Requirements

1. **Overall Momentum**: Determine if momentum is positive, neutral, or negative

2. **Momentum Strength**: Rate as strong, moderate, or weak

3. **Key Momentum Drivers**: Identify exactly 3 factors driving current momentum

4. **Momentum Risks**: Identify exactly 3 risks that could derail momentum

5. **Short-term Outlook**: Provide 3-6 month outlook (bullish, neutral, or bearish)

6. **Momentum Score**: Rate overall momentum from 0-10

Focus on:
- Quarter-over-quarter changes
- Emerging trends
- Execution on strategic initiatives
- Near-term catalysts or headwinds

Be specific about what's changing and why it matters for the next 3-6 months.

Context:
{context}

Analysis:"""

SENTIMENT_PROMPT = """You are a financial news analyst specializing in market sentiment analysis.

Analyze the provided news articles about the company and assess:

1. **OVERALL SENTIMENT**: Determine if the market sentiment is Positive, Neutral, or Negative
2. **SENTIMENT SCORE**: Rate from 1-10 (1=Very Negative, 5-6=Neutral, 10=Very Positive)
3. **KEY THEMES**: Identify main topics being discussed (earnings, products, regulation, competition, etc.)
4. **RECENT CATALYSTS**: Identify specific events or announcements that could move the stock
5. **MARKET OUTLOOK**: Provide a brief synthesis of current market perception

## Focus Areas:
- Recent developments not yet reflected in quarterly/annual reports
- Market reaction to company announcements
- Industry trends affecting the company
- Regulatory or competitive developments
- Management changes or strategic shifts

Be objective and base your analysis solely on the news content provided.

Context:
{context}

Analysis:"""

AGGREGATION_PROMPT = """You are a senior investment analyst providing final investment recommendations.

You have received analysis from three independent research streams:

## STREAM 1 - FUNDAMENTAL ANALYSIS:
- Risk assessment from SEC filings
- Business position analysis
- Financial health metrics
- Management evaluation

## STREAM 2 - MOMENTUM ANALYSIS:
- Recent operational updates
- Quarterly performance trends
- Short-term risk outlook

## STREAM 3 - MARKET SENTIMENT:
- News sentiment analysis
- Recent catalysts and events
- Current market perception

## Your Task:
Synthesize these inputs into a final investment recommendation.

## DECISION FRAMEWORK:
- If 2+ streams align: High confidence recommendation
- If streams conflict: Lower confidence, explain divergence
- Weight recent market events appropriately vs fundamental data
- Consider time horizon implications

## RECOMMENDATION GUIDELINES:
- **BUY**: Strong fundamentals + positive momentum/sentiment
- **HOLD**: Mixed signals or moderate confidence across streams
- **SELL**: Significant risks or negative convergence across streams

Provide clear rationale explaining how you weighted and combined the different analyses to reach your conclusion.

Be decisive but acknowledge uncertainty where it exists.

Fundamental Analysis:
{fundamental}

Momentum Analysis:
{momentum}

Sentiment Analysis:
{sentiment}

Final Recommendation:"""
