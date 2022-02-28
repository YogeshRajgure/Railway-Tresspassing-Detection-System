import mysql.connector


def connect():
    db = mysql.connector.connect(host='localhost',
                                 port='3306',
                                 database='railways_userdb',
                                 user='root',
                                 password='white hat')
    # cursor = connector.cursor()
    return db

# Database creation query

# CREATE DATABASE railways_userdb;
#
# USE railways_userdb;
#
# CREATE TABLE user (
#     user int(20),
#     username varchar(255),
#     password varchar(255),
#     email_id varchar(255),
#     firstname varchar(255),
#     lastname varchar(255)
# );
#
# INSERT INTO user VALUES(1, "Karthik", "karthik123", "karthik@gmail.com", "Karthik", "Arumugam");
# INSERT INTO user VALUES(2, "Yogesh", "yogesh123", "yogesh@gmail.com", "Yogesh", "Rajgure");
# INSERT INTO user VALUES(3, "Harsh", "harsh123", "harsh@gmail.com", "Harsh", "Ingle");
