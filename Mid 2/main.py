from jinja2 import Template

categories = [
    "Billing",
    "Technical",
    "Account",
    "Shipping"
]


examples = [
    {
        "text": "I was charged twice for my subscription",
        "category": "Billing"
    },
    {
        "text": "The application crashes when I upload a file",
        "category": "Technical"
    },
    {
        "text": "I cannot reset my password",
        "category": "Account"
    }
]

ticket = "My credit card was charged but my subscription is still inactive."


with open("prompt.jinja") as f:
    template = Template(f.read())

prompt = template.render(
    categories=categories,
    examples=examples,
    ticket=ticket
)

print(prompt)