import json
from faker import Faker
from faker.providers import person, internet, company, date_time, address, lorem
import random
import argparse
from tqdm import tqdm

# Lists of common first names, last names, and domain names
first_names = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen"
]

last_names = [
    "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Garcia", "Martinez", "Robinson"
]

common_domains = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "comcast.net", "live.com", "msn.com", "icloud.com", "me.com"
]

class CustomLoremProvider(lorem.Provider):
    word_list = [
        "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
        "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "papaya",
        "quince", "raspberry", "strawberry", "tangerine", "ugli", "violet",
        "watermelon", "xigua", "yellow", "zucchini"
    ]

    def word(self):
        return self.random_element(self.word_list)

class CustomEmailProvider(internet.Provider):
    def sender_email(self, is_legitimate):
        first_name = self.generator.random_element(first_names)
        last_name = self.generator.random_element(last_names)
        domain = self.generator.random_element(common_domains)

        if is_legitimate:
            email = f'{first_name.lower()}.{last_name.lower()}@{domain}'
            return f'"{first_name} {last_name}" <{email}>', first_name, last_name
        else:
            mismatched_first_name = self.generator.random_element(first_names)
            mismatched_last_name = self.generator.random_element(last_names)
            formats = [
                lambda: f'"{first_name} {last_name}" ({mismatched_first_name} {mismatched_last_name}) <{self.generator.email()}>',
                lambda: f'"{first_name} {last_name}" {mismatched_first_name} {mismatched_last_name} <{self.generator.email()}> <{self.generator.email()}>',
                lambda: f'"{first_name} {last_name}" <{self.generator.email()}>'
            ]
            return random.choice(formats)(), first_name, last_name

    def custom_discount(self):
        return random.randint(10, 50)

    def custom_promo_code(self):
        return f"{self.generator.random_uppercase_letter()}{self.generator.random_number(digits=4)}"

    def custom_event_date(self):
        return self.generator.date_this_year()

    def custom_event_location(self):
        return self.generator.address()

    def custom_invoice_number(self):
        return self.generator.random_number(digits=6)

    def custom_item(self):
        return self.random_element(CustomLoremProvider.word_list).capitalize()

    def custom_company_name(self):
        return self.generator.company()

    def custom_company(self):
        return self.custom_company_name()  # Alias for company placeholder

    def custom_due_date(self):
        return self.generator.date_this_year()

    def custom_account_info(self):
        return self.random_element(CustomLoremProvider.word_list).capitalize()

    def custom_project_name(self):
        return self.generator.bs().title()

    def custom_client_name(self):
        return self.generator.name()

    def custom_topic(self):
        return self.generator.bs()

    def custom_meeting_date(self):
        return self.generator.date_time_this_year()

    def custom_meeting_time(self):
        return self.generator.time()

    def custom_order_number(self):
        return self.generator.random_number(digits=8)

    def custom_fake_name(self):
        return self.generator.name()

    def custom_full_name(self):
        return f"{self.generator.first_name()} {self.generator.last_name()}"

    def custom_name(self):
        return self.generator.name()

    def custom_signature(self):
        return self.generator.name()

    def subject_and_content(self, templates, is_legitimate, recipient_name, sender_name):
        if is_legitimate:
            email_type = random.choice([key for key in templates.keys() if key != "fraudulent"])
        else:
            email_type = "fraudulent"

        template = random.choice(templates[email_type])
        placeholders = {ph: getattr(self, f'custom_{ph}')() for ph in template.get("placeholders", [])}
        placeholders["name"] = recipient_name
        placeholders["signature"] = sender_name

        try:
            subject = template["subject"].format(**placeholders)
            content = template["content"].format(**placeholders)
        except KeyError as e:
            print(f"KeyError: Missing placeholder {e} in template {template}")
            raise

        return subject, content

def generate_single_email(fake, templates):
    is_legitimate = random.choices([True, False], weights=[0.8, 0.2], k=1)[0]
    email_from, sender_first_name, sender_last_name = fake.sender_email(is_legitimate)

    to_first_name = fake.random_element(first_names)
    to_last_name = fake.random_element(last_names)
    to_domain = fake.random_element(common_domains)
    to_email = f'{to_first_name.lower()}.{to_last_name.lower()}@{to_domain}'
    recipient_name = f"{to_first_name} {to_last_name}"
    cc_email = fake.email() if random.random() > 0.7 else ""

    subject, content = fake.subject_and_content(templates, is_legitimate, recipient_name, f"{sender_first_name} {sender_last_name}")

    max_attempts = 5
    attempts = 0
    while email_from.split()[-1] == f'<{to_email}>' and attempts < max_attempts:
        email_from, sender_first_name, sender_last_name = fake.sender_email(is_legitimate)
        attempts += 1

    if email_from.split()[-1] == f'<{to_email}>':
        raise Exception("Failed to generate a unique sender email different from recipient email")

    email = {
        "subject": subject,
        "from": email_from,
        "date": fake.date_time_this_year().strftime("%A, %B %d, %Y %I:%M %p"),
        "to": to_email,
        "cc": cc_email,
        "content": content,
        "label": "legitimate" if is_legitimate else "fraudulent"
    }
    return email

def generate_email_batch(num_emails, templates):
    fake = Faker()
    fake.add_provider(person.Provider)
    fake.add_provider(company.Provider)
    fake.add_provider(date_time.Provider)
    fake.add_provider(address.Provider)
    fake.add_provider(CustomLoremProvider)
    fake.add_provider(CustomEmailProvider)

    email_batch = [generate_single_email(fake, templates) for _ in tqdm(range(num_emails), desc="Generating emails")]
    return email_batch

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        for email in data:
            json_file.write(json.dumps(email) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic email data")
    parser.add_argument('-n', '--num_emails', type=int, default=50, help='Number of emails to generate')
    parser.add_argument('-o', '--output', type=str, default='synthetic_emails.json', help='Output file name')

    args = parser.parse_args()

    num_emails = args.num_emails
    output_file = args.output

    with open('email_templates.json', 'r') as f:
        templates = json.load(f)

    email_data = generate_email_batch(num_emails, templates)
    save_to_json(email_data, output_file)
