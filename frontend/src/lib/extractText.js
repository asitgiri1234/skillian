/**
 * Turn a .pdf or .docx File into plain text, in the browser.
 *
 * Why this exists: `POST /resumes` is `application/json` and takes `raw_text`.
 * The backend has no PDF or DOCX handling at all, and it is closed to further
 * changes, so extraction has to happen here.
 *
 * The upside is that the scanned-PDF case gets a *better* error than a server
 * could give: we can see that a PDF has pages but no extractable text and say
 * exactly that, rather than posting an empty string and getting back a generic
 * validation failure.
 */

import mammoth from 'mammoth'
import * as pdfjs from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

// Vite serves the worker as a URL asset; without this pdf.js tries to load a
// worker from a path that does not exist in the dev server and silently falls
// back to running on the main thread (or fails outright in a build).
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorker

export const MAX_FILE_BYTES = 5 * 1024 * 1024
export const ACCEPTED_EXTENSIONS = ['.pdf', '.docx']

/** Below this many characters, a "successful" extraction is not a resume. */
const MIN_USEFUL_CHARS = 120

export class ExtractionError extends Error {
  /** @param {'file_type'|'file_size'|'empty'|'image_only'|'corrupt'} kind */
  constructor(message, kind, hint = null) {
    super(message)
    this.name = 'ExtractionError'
    this.kind = kind
    this.hint = hint
  }
}

function extensionOf(name) {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot).toLowerCase()
}

export function validateFile(file) {
  const ext = extensionOf(file.name)
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    throw new ExtractionError(
      `${file.name} is not a supported file type.`,
      'file_type',
      `Upload a ${ACCEPTED_EXTENSIONS.join(' or ')} file. Export from Word or Google Docs if you have another format.`,
    )
  }
  if (file.size > MAX_FILE_BYTES) {
    const mb = (file.size / 1024 / 1024).toFixed(1)
    throw new ExtractionError(
      `${file.name} is ${mb}MB, over the 5MB limit.`,
      'file_size',
      'Most resumes are well under 1MB. A large file usually means embedded images — try exporting again as text-based PDF.',
    )
  }
  return ext
}

async function extractPdf(file) {
  const buffer = await file.arrayBuffer()
  let doc
  try {
    doc = await pdfjs.getDocument({ data: buffer }).promise
  } catch (cause) {
    throw new ExtractionError(
      `${file.name} could not be opened as a PDF.`,
      'corrupt',
      'The file may be damaged or password-protected. Try re-exporting it.',
    )
  }

  const pages = []
  for (let n = 1; n <= doc.numPages; n += 1) {
    const page = await doc.getPage(n)
    const content = await page.getTextContent()
    // Join with spaces, then let line structure come from the item positions:
    // pdf.js emits one item per text run, and runs on the same line share a
    // transform Y. Section markers in the backend anchor to line starts, so
    // flattening everything to one line would cost requirement detection.
    let lastY = null
    let line = []
    for (const item of content.items) {
      const y = item.transform?.[5]
      if (lastY !== null && y !== undefined && Math.abs(y - lastY) > 2) {
        pages.push(line.join(' '))
        line = []
      }
      if (item.str) line.push(item.str)
      if (y !== undefined) lastY = y
    }
    if (line.length) pages.push(line.join(' '))
  }

  const text = pages.join('\n').replace(/[ \t]+/g, ' ').trim()

  if (text.length < MIN_USEFUL_CHARS) {
    // The scanned-resume case. A photographed or scanned CV is a PDF whose
    // pages are images; there is no text layer to extract, and posting the
    // empty result would surface as a confusing 422 from the parser.
    throw new ExtractionError(
      `No readable text found in ${file.name}.`,
      'image_only',
      `This looks like a scanned or image-only PDF — it has ${doc.numPages} page(s) but no text layer. Run it through OCR, or export a text-based PDF from your word processor.`,
    )
  }
  return text
}

async function extractDocx(file) {
  const buffer = await file.arrayBuffer()
  let result
  try {
    result = await mammoth.extractRawText({ arrayBuffer: buffer })
  } catch (cause) {
    throw new ExtractionError(
      `${file.name} could not be opened as a Word document.`,
      'corrupt',
      'The file may be damaged, or it may be an older .doc rather than .docx. Re-save it as .docx.',
    )
  }
  const text = (result.value ?? '').replace(/[ \t]+/g, ' ').trim()
  if (text.length < MIN_USEFUL_CHARS) {
    throw new ExtractionError(
      `No readable text found in ${file.name}.`,
      'empty',
      'The document appears to be empty, or its content is entirely inside images or text boxes.',
    )
  }
  return text
}

/** @returns {Promise<string>} plain text; throws ExtractionError otherwise. */
export async function extractText(file) {
  const ext = validateFile(file)
  return ext === '.pdf' ? extractPdf(file) : extractDocx(file)
}
