# Importação das bibliotecas necessárias
import json 
import os
from datetime import datetime

import yfinance as yf
import streamlit as st

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

# Função para coletar os dados do Yahoo Finance
def fectch_stock_history_price(ticket):
    stock = yf.download(ticket, start="2023-01-01", end="2023-12-31")
    return stock

# Criação da ferramenta para coletar os dados do Yahoo Finance
yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "This tool fetches stocks prices for {ticket} from some year about a specific stock from Yahoo Finance API.",
    func= lambda ticket: fectch_stock_history_price(ticket)
)

# Criação do modelo de LLM usando OpenAI
load_dotenv() # Carrega as variáveis de ambiente
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Criação do Agente que irá analisar os preços das ações
stockPriceAnalyst = Agent(
  role  = "Senior Stock Price Analyst",
  goal = "Find the {ticket} stock prices and analyses future prices and trends.",
  backstory = "You're a highly experienced stock price analyst with a deep understanding of the stock market and good prediction skills about this market.",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  Tools = [yahoo_finance_tool],
  allow_delegation = False
)

# Criação da tarefa para analisar os preços das ações
getStockPrice = Task(
  description = "Analyse the stock {ticket} price history, create a trend analyses of up, down or sideways and predict future prices and trends.",
  expected_output = """Specify the current trend of the stock price and predict the future price and trend.
  eg. stock= 'APPL, price UP'
  """,
  agent = stockPriceAnalyst,
)

# Importação da ferramenta para coletar os resultados da busca no DuckDuckGo, ela já existe por isso não é necessário a criação
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10) 

# Criação do Agente que irá analisar as notícias sobre as ações
newsAnalyst = Agent(
  role  = "Stock News Analyst",
  goal = "In a Summary find the latest news about {ticket} stock and analyse how it will impact the stock price. Specify if it will go up, down or sideways. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
  backstory = """You're a highly experienced stock news analyst with a deep understanding of the stock market and good prediction skills about this market. 
  You have access to the latest news and information about the stock market and you can analyse how it will impact the stock price.
  You're also master level analysts in the tradicional markets and have deep understanding of human psychology and how it impacts the stock market.
  You understand news, theirs tittles and information, but you look at those with a health dose of skepticism and you know that not all news are true or have a real impact on the stock market.
  You also consider the source of the news and the credibility of the source.
  """,
  verbose = True,
  llm = llm,
  max_iter = 10,
  memory = True,
  Tools = [search_tool],
  allow_delegation = False
)

# Criação da tarefa para analisar as notícias sobre as ações
getNews = Task(
  description = f"""Take the stock and always include BTC to it (if not request).
  Use the search tool to search each one individually.
  The current date is {datetime.now()}.
  compose the results into a helpfull report
  """,
  expected_output = """A summary of the overall market and one sentence summary for each requested asset.
  Include a fear/greed score for each asset based on the news. Use the format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
  agent = newsAnalyst
)

# Criação do agente que analisa os dados das ações
stockAnalystWrite = Agent(
  role = "Senior Stock Analyst Writer",
  goal = "Analyze the trends price and news to write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.",
  backstory = """You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling sotires and narratives that resonate with wider audiences
  
  You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
  You're abe to hold multiple opinions when analyzing anything.
  """,
  verbose = True,
  llm= llm,
  max_iter = 5,
  memory = True,
  allow_delegation = True
)

# Criação da tarefa Principal
writeAnalyses = Task(
  description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company that is brief and highlights the most important points.
  Focus on the stock price trend, news and fear/greed score. what are the near future considerations?
  Include the previous analyses of stock trend and news summary.
  """,
  expected_output = """An eloquent 3 paragraphs newsletter fornated as markdown in an easy readable manner. It should contain:
  
  - 3 Bullets executive summary
  - Introduction - set the overall picture and spike up the interest
  - main part provides the meat of the analysis including the news summary and fear/gred scores
  -summary - key facts and concrete future trend prediction - up, down or sideways.
  """,
  agent = stockAnalystWrite,
  context = [getStockPrice, getNews]
)


# Criação da Crew para organização e centralização das tarefas
crew = Crew(
  agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
  tasks = [getStockPrice, getNews, writeAnalyses],
  verbose = 2,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = llm,
  max_iter = 15,
  )

with st.sidebar:
    st.title("CrewAI Stocks")
    st.subheader("Welcome to CrewAI Stocks! Here you can analyze the stock prices and news to predict future trends and write a newsletter about it.")
    st.write("Enter the Stock to Research")

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label='Run Research')

if submit_button:
    if not topic:
      st.error("Please fill the ticket field.")
    else:
      result = crew.kickoff(inputs={"ticket": topic})
      st.subheader("The analysis is complete! Here is the newsletter:")
      st.write(result['final_output'])