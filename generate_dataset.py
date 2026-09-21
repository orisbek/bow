#!/usr/bin/env python3
"""Generate large English SMS dataset."""

import random

# English ham messages (normal SMS)
ham_templates = [
    "Hi, how are you?",
    "How's work going? Busy day?",
    "When will you come home?",
    "Are you home?",
    "Call me when you can",
    "Let's meet this weekend",
    "Thanks for your help!",
    "When can we meet?",
    "How's the family? Everything good?",
    "Say hi to everyone",
    "Come to the party on Friday",
    "I'm at the store, need anything?",
    "Don't forget to buy milk",
    "Let's eat together this weekend",
    "I miss you",
    "How was work today?",
    "Thanks for the invitation",
    "Did you watch the latest movie?",
    "Want to play tennis?",
    "Meeting at 3pm, don't forget",
    "How are you feeling? Get better soon",
    "Don't forget to return my book",
    "When will you be home?",
    "Calling, pick up the phone",
    "Waiting for you at the entrance",
    "Just got home, I'm tired",
    "Tomorrow is your birthday, remember?",
    "How's school going?",
    "Need help with homework?",
    "Got your letter",
    "Want to talk to you",
    "Don't be mad at me",
    "Can we make up?",
    "I'm telling the truth",
    "I love you",
    "Always here for you",
    "I'm worried about you",
    "Take care of yourself",
    "Have a nice day",
    "See you later",
    "Bye bye",
    "Kiss kiss",
    "Hug you",
    "You're so funny",
    "You're cool!",
    "Great guy",
    "Beautiful girl",
    "Best friend",
    "Loyal buddy",
    "Reliable person",
    "Good person",
]

# English spam messages
spam_templates = [
    "CONGRATULATIONS! You won 1 million dollars!",
    "Congrats! Claim your prize NOW",
    "Click here for instant prize: bit.ly/win",
    "URGENT: Update your information immediately",
    "Your account has been locked! Verify now",
    "Free credit! Approval guaranteed 99%",
    "Get 50000 dollars in 5 minutes",
    "Click here to win now",
    "AMAZING DEAL: Cheap internet service",
    "Limited offer! Hurry up today",
    "Make money online! Easy and fast",
    "Work from home, earn 50k monthly",
    "Lose weight in one week!",
    "Miracle cure for all diseases",
    "English courses only 99 rubles",
    "Free programming training courses",
    "Investment: double your money fast",
    "Online casino: win real money",
    "Play poker: earn money daily",
    "Sports betting: guaranteed wins",
    "Instant loans up to 500000 rubles",
    "Personal loan no credit check",
    "Refinance your loans today",
    "Life insurance best rates",
    "Real estate at half price",
    "Dream apartment waiting for you",
    "Turkey vacation on sale",
    "Car loans zero down payment",
    "Latest mobile phone available",
    "Laptop with 70% discount",
    "Smartphone mega sale",
    "Tablet delivery to your home",
    "Premium headphones cheap",
    "Smart watch latest model",
    "Professional digital camera",
    "Ultra HD 4K television",
    "Energy saving refrigerator",
    "Automatic washing machine",
    "Apartment furniture discount",
    "Luxury cosmetics sale",
    "Original perfume cheap",
    "Home beauty services",
    "Beauty salon near metro",
    "Professional nails and pedicure",
    "Famous barber haircut",
    "Fitness gym maximum discount",
    "Yoga and meditation online",
    "Beginner dance classes",
    "Painting lessons available",
    "Music lessons and instruments",
    "Dating: beautiful girls waiting",
    "Foreign dating site",
    "Online love guaranteed",
    "Grocery delivery service",
    "Pizza home delivery",
    "Sushi and rolls special offer",
    "Flower delivery same day",
    "Taxi 50% discount",
    "Car rental cheap rates",
    "Driving school training",
    "Car repair with warranty",
    "Auto maintenance service",
    "Tire and wheel service",
    "Best dentist available",
    "Painless dentistry",
    "Dental prosthetics",
    "Orthodontic braces",
    "Professional teeth cleaning",
    "Eye doctor consultation",
    "Glasses and contact lenses",
    "Laser eye surgery",
    "Therapeutic massage",
    "Osteopathy treatment",
    "Acupuncture therapy",
    "Homeopathic remedies",
    "Herbal medicine treatment",
    "Aromatherapy essential oils",
    "Relaxation meditation",
    "Yoga classes available",
]

ham_extensions = [
    ", let's meet",
    ", when are you free?",
    ", don't forget!",
    ", thanks!",
    ", love you",
    ", hugs",
    ", bye bye",
    ", see you later",
    ", huge hug",
    ", miss you",
    ", how are you?",
    ", need help?",
    ", come over",
    ", call me",
    ", text me",
    ", reply please",
    ", don't be silent",
    ", tell me the truth",
    ", trust me",
    ", believe in me",
    ", you can count on me",
    ", by your side",
    ", waiting for you",
    ", love very much",
    ", don't be sad",
    ", smile please",
    ", so funny",
    ", awesome!",
    ", great!",
    ", perfect!",
    ", thank you so much",
    ", thanks a lot",
]

spam_extensions = [
    " Go to: link.ru",
    " Link: bit.ly/promo",
    " Activate: click.com",
    " Deadline tomorrow!",
    " Hurry before today ends!",
    " Only limited quantities!",
    " Exclusive for you!",
    " Exclusive offer!",
    " Don't wait!",
    " Be quick!",
    " Rush!",
    " Sale! Sale! Sale!",
    " Discount 50%!",
    " Discount 70%!",
    " Discount 90%!",
    " Free!",
    " Easy!",
    " Fast!",
    " Simple!",
    " Guaranteed!",
    " Proven!",
    " Great reviews!",
    " Highly recommend!",
    " Check reviews!",
    " See evidence!",
]

def generate_dataset(count=5000):
    """Generate English SMS dataset."""
    lines = []
    
    # Generate ham messages (70% of dataset)
    ham_count = int(count * 0.7)
    for _ in range(ham_count):
        template = random.choice(ham_templates)
        extension = random.choice(ham_extensions) if random.random() > 0.3 else ""
        message = template + extension
        lines.append(f"ham\t{message}")
    
    # Generate spam messages (30% of dataset)
    spam_count = count - ham_count
    for _ in range(spam_count):
        template = random.choice(spam_templates)
        extension = random.choice(spam_extensions) if random.random() > 0.4 else ""
        message = template + extension
        lines.append(f"spam\t{message}")
    
    # Shuffle
    random.shuffle(lines)
    
    return "\n".join(lines)

if __name__ == "__main__":
    random.seed(42)
    dataset = generate_dataset(5500)
    
    with open("data/SMSSpamCollection", "w", encoding="utf-8") as f:
        f.write(dataset)
    
    print(f"Generated 5500 English SMS messages")
    print("Dataset saved to data/SMSSpamCollection")
