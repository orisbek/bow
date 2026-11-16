"""New, manually labelled control texts, separate from model selection."""

CONTROL_TEXTS = [
    {
        "id": "01",
        "title": "Встреча после занятий",
        "text": "Hey, our study group is meeting at the library at six. Can you bring your notes?",
        "expected": "ham",
    },
    {
        "id": "02",
        "title": "Сообщение о доставке",
        "text": "Your parcel is at reception. I can bring it upstairs when I finish my lunch break.",
        "expected": "ham",
    },
    {
        "id": "03",
        "title": "Обещание денежного приза",
        "text": "WINNER! Your mobile number has won a FREE cash prize of 2000 pounds. Call 09061234567 now to claim!",
        "expected": "spam",
    },
    {
        "id": "04",
        "title": "Платная подписка на мелодии",
        "text": "Get FREE ringtones today! Text TONE to 87654 to subscribe. Only 3 pounds per week. Reply STOP to cancel.",
        "expected": "spam",
    },
    {
        "id": "05",
        "title": "Слово free в личной переписке",
        "text": "Are you free on Saturday? We could cook dinner together and watch that film you mentioned.",
        "expected": "ham",
    },
]
