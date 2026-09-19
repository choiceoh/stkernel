#!/usr/bin/env bash
# Restart the mail answers with thinking on for all and the real email-analysis skill in half the analyses.
cd ~/q38mtp-gen2
pkill -f "selfgen3.py answer v3 v3/mail_prompts.jsonl" && sleep 2
rm -f v3/mail_answers.jsonl v3/mail_answer.log
python3 mailgen3.py v3 600
(nohup bash selfgen3.sh answer v3 v3/mail_prompts.jsonl v3/mail_answers.jsonl > v3/mail_answer.log 2>&1 < /dev/null &)
sleep 30
tail -n 1 v3/mail_answer.log | cut -c1-160
