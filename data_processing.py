import pandas as pd
import sqlalchemy as sa


engine = sa.create_engine('mysql+pymysql://root:Admin@localhost/yelp_db?charset=utf8')
conn = engine.connect()
if conn:
    sql_review = "select  text, stars  from review"
    data_review = pd.read_sql(sql=sql_review, con=conn)
    data = data_review.to_json(orient='records')
