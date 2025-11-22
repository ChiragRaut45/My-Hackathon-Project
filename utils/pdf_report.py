from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

def generate_pdf_report(filename, patient_id, raw_inputs, scaled_inputs, prediction, probabilities):
    """
    Creates a PDF diagnostic report.
    """

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "MediGuard AI — Diagnostic Report")
    y -= 40

    # Patient info
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Patient ID: {patient_id}")
    y -= 20
    c.drawString(50, y, "Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    y -= 30

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Prediction: {prediction}")
    y -= 30

    # Probabilities
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Probability Distribution:")
    y -= 20

    c.setFont("Helvetica", 10)
    for disease, prob in probabilities.items():
        c.drawString(60, y, f"{disease}: {prob:.4f}")
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 50

    y -= 20

    # Raw Inputs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Raw Clinical Inputs:")
    y -= 20

    c.setFont("Helvetica", 10)
    for k, v in raw_inputs.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    y -= 20

    # Scaled Inputs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Scaled Inputs (0–1):")
    y -= 20

    c.setFont("Helvetica", 10)
    for k, v in scaled_inputs.items():
        c.drawString(60, y, f"{k}: {v:.4f}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    return filename
