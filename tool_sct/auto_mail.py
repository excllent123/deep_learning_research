import argparse 
import smtplib

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('-h', '--host')
    cli.add_argument('-n', '--name')
    cli.add_argument('-p', '--password')    

    cli.add_argument('-f', '--from')
    cli.add_argument('-t', '--to')
    cli.add_argument('-s', '--subject')
    cli.add_argument('-x', '--text')

SERVER = "smtp.google.com"
FROM = "johnDoe@gmail.com"
TO = ["JaneDoe@gmail.com"] # must be a list

SUBJECT = "Hello!"
TEXT = "This is a test of emailing through smtp in google."

# Prepare actual message
message = """From: %s\r\nTo: %s\r\nSubject: %s\r\n\

%s
""" % (FROM, ", ".join(TO), SUBJECT, TEXT)

# Send the mail

server = smtplib.SMTP(SERVER)
server.login("MrDoe", "PASSWORD")
server.sendmail(FROM, TO, message)
server.quit()