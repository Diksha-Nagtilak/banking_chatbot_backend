"""
Banking Knowledge Base for RAG (Retrieval-Augmented Generation).
Contains clean, structured documents representing banking products, policies, and FAQs.
"""
from app.schemas import KnowledgeDocument

BANKING_KNOWLEDGE_BASE = [
    KnowledgeDocument(
        id="DOC_001",
        title="Checking & Savings Account Overview",
        category="account_services",
        content=(
            "AegisBank offers Checking and High-Yield Savings accounts. Checking accounts are designed "
            "for daily debit card transactions, ATM withdrawals, and bill payments with zero monthly maintenance "
            "fees when enrolled in direct deposit. High-Yield Savings accounts currently earn an APY of 4.75% "
            "with interest compounded daily. Both account types are FDIC-insured up to $250,000 per depositor. "
            "Customers can check real-time balances, view monthly statements, and set low-balance alert thresholds."
        ),
    ),
    KnowledgeDocument(
        id="DOC_002",
        title="Funds Transfer Limits, Methods & Timelines",
        category="funds_transfer",
        content=(
            "Customers can transfer money through three primary methods: "
            "1. Domestic ACH Transfers: Free of charge, processed within 1 to 3 business days. Daily limit: $5,000. "
            "2. Wire Transfers: Same-day delivery for requests submitted before 4:00 PM EST. Domestic wires carry a $20 fee, "
            "while international wires carry a $40 fee. Daily wire limit: $50,000. "
            "3. Instant Peer-to-Peer (P2P): Real-time transfers using a recipient's phone number or email address with zero fees. "
            "Daily P2P limit: $2,500. All transfers require account verification and fraud monitoring."
        ),
    ),
    KnowledgeDocument(
        id="DOC_003",
        title="Debit & Credit Card Security and Freeze Procedures",
        category="cards_and_security",
        content=(
            "If a customer loses their card or suspects fraud, they can immediately execute an Emergency Card Freeze. "
            "Freezing a card instantly blocks all new in-store transactions, online purchases, and ATM cash withdrawals, "
            "while recurring authorized utility bills remain uninterrupted. Cards can be unfrozen instantly when found. "
            "If a card is permanently stolen, customer support will issue a free replacement card with a new 16-digit number "
            "dispatched via expedited 2-day delivery. PIN changes can be completed securely via the mobile app."
        ),
    ),
    KnowledgeDocument(
        id="DOC_004",
        title="Mortgage, Personal Loan & APR Interest Rates",
        category="loans_and_credit",
        content=(
            "AegisBank financing options and APR rates: "
            "• 30-Year Fixed Mortgage: 6.45% APR (requires minimum 3% down payment, 620 credit score). "
            "• 15-Year Fixed Mortgage: 5.75% APR (lower total interest cost over loan lifetime). "
            "• Personal Unsecured Loans: Starting at 7.99% APR for amounts between $2,000 and $50,000 with flexible 24 to 60-month terms. "
            "• New & Used Auto Loans: Starting from 5.25% APR for terms up to 72 months. "
            "APR (Annual Percentage Rate) includes the nominal interest rate plus all origination fees, representing the true annual cost of borrowing."
        ),
    ),
    KnowledgeDocument(
        id="DOC_005",
        title="Transaction Disputes and Zero Liability Protection",
        category="cards_and_security",
        content=(
            "Under the AegisBank Zero Liability Guarantee, customers are not responsible for unauthorized charges "
            "reported promptly. To dispute an unrecognized charge: "
            "1. Freeze the card immediately to prevent further charges. "
            "2. File a dispute claim within 60 calendar days of the statement issue date. "
            "3. Provisional credit is typically deposited into the customer's checking account within 2 business days "
            "while the fraud investigation is actively conducted. Investigations are finalized within 10 to 45 business days."
        ),
    ),
    KnowledgeDocument(
        id="DOC_006",
        title="Bank Routing Transit Numbers, SWIFT & Customer Support",
        category="general_banking_faq",
        content=(
            "Official AegisBank Institutional Identifiers: "
            "• Domestic Routing Transit Number (ABA): 021000021 (used for direct deposits, payroll, and ACH debits). "
            "• International SWIFT / BIC Code: AEGSUS33XXX (used for international inbound wires). "
            "• 24/7 Telephone Customer Support: 1-800-555-AEGIS (1-800-555-2344). "
            "• Branch Lobby Hours: Monday – Friday: 9:00 AM – 5:00 PM | Saturday: 9:00 AM – 1:00 PM EST."
        ),
    ),
    KnowledgeDocument(
        id="DOC_007",
        title="Savings Strategies, 50/30/20 Rule & Compound Interest",
        category="account_services",
        content=(
            "Financial Education & Wealth Building: "
            "• The 50/30/20 Budgeting Rule suggests allocating 50% of after-tax income to Needs (housing, groceries, utilities), "
            "30% to Wants (dining, hobbies, subscriptions), and 20% to Savings and debt repayment. "
            "• Emergency Fund: Financial planners recommend holding 3 to 6 months of essential living expenses in a liquid High-Yield Savings Account. "
            "• Compound Interest: Earning interest on previously accumulated interest. Formula: A = P(1 + r/n)^(nt). "
            "The Rule of 72 estimates years to double money: divide 72 by the annual interest rate (e.g. 72 / 8% = 9 years to double)."
        ),
    ),
]
