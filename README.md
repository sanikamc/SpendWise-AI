# SpendWise AI — Smart Expense Anomaly Detector

SpendWise AI is a web-based personal expense management application that helps users track spending, manage budgets, view financial summaries, and identify potentially unusual transactions using machine learning.

## Features

- **User Authentication:** User registration, login, and logout.
- **Expense Management:** Add, edit, and delete expenses.
- **Budget Management:** Set category-based budgets and review spending.
- **AI Anomaly Detection:** Identify potentially unusual transactions using an Isolation Forest model.
- **Dashboard and Analytics:** Review expense summaries and spending patterns.
- **Reports and Exports:** Export financial reports to CSV and Excel.
- **Data Isolation:** Keep each user's expenses and budgets separate.
- **Demo Data:** Support for sample data to demonstrate application functionality.

## Technology Stack

- **Backend:** Python, Flask
- **Database:** SQLite, Flask-SQLAlchemy
- **Machine Learning:** scikit-learn, Isolation Forest
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **Testing:** pytest

## Project Structure

```text
Smart_Expense_Anomaly_Detector/
├── app.py
├── config.py
├── database.py
├── model.py
├── requirements.txt
├── .gitignore
├── scripts/
├── static/
├── templates/
└── tests/
    ├── test_app.py
    ├── test_auth.py
    └── test_model.py
```

## Getting Started

### Prerequisites

- Python 3.11 or a compatible Python version
- pip
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sanikamc/SpendWise-AI.git
   cd SpendWise-AI
   ```

2. Create and activate a virtual environment.

   **Windows PowerShell:**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Configure any required environment variables using your local configuration. Never commit passwords, API tokens, or other secrets.

5. Start the application using the project's configured Flask entry point. For this project, the entry point is `app.py`:

   ```powershell
   python app.py
   ```

6. Open the local URL displayed in the terminal.

## Running Tests

Run the automated test suite from the project root:

```powershell
python -m pytest -v
```

### Latest Test Results

- **27 tests passed**
- **0 tests failed**
- Execution time: approximately 15.65 seconds

The latest run reported 97 deprecation and compatibility warnings. These should be reviewed in future maintenance.

## Machine Learning Notes

SpendWise AI uses Isolation Forest to flag transactions that may be unusual relative to the available data. An anomaly flag is an indication for review, not proof of fraud or an incorrect transaction. Results depend on the data available and the model configuration.

## Data and Security

- Local database files and backups should not be committed.
- Keep `.env` files and credentials private.
- Use synthetic data for public demonstrations.
- Do not upload real personal financial information.
- Machine-learning results should be treated as informational, not financial advice.

## Project Status

The application has passed its current automated test suite. Additional manual testing and review may be performed before final academic submission.

## Author

**Sanika M. C.**

Developed as an academic mini-project for Computer Science and Engineering.
