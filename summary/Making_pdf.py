# Python program to create
# a pdf file


from fpdf import FPDF


# save FPDF() class into a
# variable pdf
text='Indian army officer Major General D. Samanwar told NDTV news \
channel the incident was tantamount to a ceasefire violation.'
# add another cell
def pdf_write(text):
    pdf = FPDF()

# Add a page
    pdf.add_page()

# set style and size of font
# that you want in the pdf
    pdf.set_font("Arial", size = 20)

# create a cell
    pdf.cell(200, 10, txt ="Update Summarizer",
		ln = 1, align ='L')
    pdf.set_font("Arial", size = 12)
    for lines in text.splitlines():
        print(lines)
        pdf.cell(200, 10, txt =lines,align='L')
    #pdf.cell(200, 10, txt = text, ln = 2, align = 'C')
    pdf.output("updateSummarizer.pdf")
    print("pdf created")
    return
pdf_write(text)
#pdf.cell(200, 10, txt = "A Computer Science portal for geeks.",
	#	ln = 2, align = 'C')

# save the pdf with name .pdf
#pdf.output("GFG.pdf")
