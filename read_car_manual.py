from manual_qa import ManualQA

manual = ManualQA(
    pdf_path="/home/jetson-1/Rowdy_chatbot/GR86 user manual.pdf",
    cache_dir=".manual_cache",
    model="gpt-4.1-mini",
)

res = manual.ask("What is the wheel lug torque?")
print(res["answer"])         # includes (p. N)
print(res["pages_used"])     # list of exact pages used