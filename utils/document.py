import os
from datetime import datetime
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
import re
from typing import Optional, List, Union, Dict, Any

def clean_content(content: Union[str, Dict, List]) -> str:
    """
    Clean and format the content for the document.
    
    Args:
        content: The raw content to clean and format. Can be a string, dictionary, or list.
        
    Returns:
        str: The cleaned and formatted content as a string.
        
    Raises:
        ValueError: If the content cannot be properly formatted.
    """
    if not content:
        return ""
        
    try:
        # If content is a string representation of an object, try to extract the actual content
        if isinstance(content, str):
            # Try to extract content from AgentHistoryList format
            if "extracted_content=" in content:
                matches = re.findall(r"extracted_content='(.*?)'", content)
                if matches:
                    # Get the last successful extraction
                    content = matches[-1]
            
            # Try to extract content from model outputs
            if "all_model_outputs" in content:
                match = re.search(r"'text': '(.*?)'", content)
                if match:
                    content = match.group(1)
        
        # Handle JSON content
        if isinstance(content, str) and (content.startswith('{') or content.startswith('[')):
            try:
                import json
                data = json.loads(content)
                if isinstance(data, dict):
                    # Format the content in a readable way
                    formatted = []
                    for key, value in data.items():
                        if isinstance(value, list):
                            formatted.append(f"\n{key.capitalize()}:")
                            for item in value:
                                if isinstance(item, dict):
                                    # Handle nested dictionaries
                                    nested = []
                                    for k, v in item.items():
                                        if isinstance(v, str):
                                            nested.append(f"{k.capitalize()}: {v}")
                                    formatted.append("- " + "; ".join(nested))
                                else:
                                    formatted.append(f"- {item}")
                        elif isinstance(value, dict):
                            # Handle nested dictionaries
                            nested = []
                            for k, v in value.items():
                                if isinstance(v, str):
                                    nested.append(f"{k.capitalize()}: {v}")
                            formatted.append(f"\n{key.capitalize()}: " + "; ".join(nested))
                        else:
                            formatted.append(f"\n{key.capitalize()}: {value}")
                    content = "\n".join(formatted)
                elif isinstance(data, list):
                    # Format list items
                    formatted = []
                    for item in data:
                        if isinstance(item, dict):
                            # Handle dictionaries in list
                            nested = []
                            for k, v in item.items():
                                if isinstance(v, str):
                                    nested.append(f"{k.capitalize()}: {v}")
                            formatted.append("- " + "; ".join(nested))
                        else:
                            formatted.append(f"- {item}")
                    content = "\n".join(formatted)
            except json.JSONDecodeError:
                # If JSON parsing fails, continue with the original content
                pass
        
        # Clean up any markdown or special characters
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Remove italic
        content = re.sub(r'`(.*?)`', r'\1', content)        # Remove code blocks
        
        # Convert markdown links to HTML links
        content = re.sub(r'\[(.*?)\]\(([^)]+)\)', r'<a href="\2">\1</a>', content)
        
        # Clean up any remaining escape characters
        content = content.replace('\\n', '\n')
        content = content.replace('\\"', '"')
        content = content.replace("\\'", "'")
        
        # Remove any emoji or special characters at the start
        content = re.sub(r'^[\U0001F000-\U0001F9FF\s]*', '', content)
        
        # Ensure proper spacing between sections
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
        
    except Exception as e:
        raise ValueError(f"Error cleaning content: {str(e)}")

def save_result_as_docx(
    prompt: str,
    result: Union[str, Dict, List],
    logs: List[str],
    filename: Optional[str] = None,
    format_prompt: Optional[str] = None
) -> str:
    """
    Save the browser agent's result and logs as a Word document.
    
    Args:
        prompt: The prompt that was given to the agent.
        result: The result returned by the agent.
        logs: List of log messages from the task execution.
        filename: Optional filename for the document. If not provided, one will be generated.
        format_prompt: Optional prompt for formatting the document.
        
    Returns:
        str: Path to the saved document.
        
    Raises:
        ValueError: If the document cannot be created or saved.
        IOError: If there are issues with file operations.
    """
    try:
        # Create a new Document
        doc = Document()
        
        # Apply base formatting
        doc = format_document(doc)
        
        # Add a title
        title = doc.add_heading('Job Search Agent Result', 0)
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_paragraph = doc.add_paragraph(f'Generated on: {timestamp}')
        time_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
        
        # Add horizontal line
        doc.add_paragraph('_' * 80)
        
        # Add prompt section
        doc.add_heading('Search Query', level=1)
        prompt_para = doc.add_paragraph()
        prompt_para.add_run(prompt)
        
        # Add format prompt section if provided
        if format_prompt:
            doc.add_heading('Format Instructions', level=1)
            format_para = doc.add_paragraph()
            format_para.add_run(format_prompt)
        
        # Add result section
        doc.add_heading('Search Results', level=1)
        
        # Clean and format the result content
        cleaned_content = clean_content(result)
        
        # Split content into paragraphs and add to document
        paragraphs = cleaned_content.split('\n')
        current_list = None
        current_heading_level = 1
        
        for para_text in paragraphs:
            if not para_text.strip():
                continue
                
            # Check if this is a heading (starts with #)
            heading_match = re.match(r'^(#+)\s+(.+)$', para_text)
            if heading_match:
                current_list = None  # Reset list context
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                doc.add_heading(heading_text, level=min(level, 3))
                current_heading_level = level
                continue
                
            # Check if this is a list item
            if para_text.startswith('- '):
                if not current_list or current_list.style.name != 'List Bullet':
                    current_list = doc.add_paragraph(style='List Bullet')
                else:
                    current_list = doc.add_paragraph(style='List Bullet')
                current_list.text = para_text[2:].strip()
                
            # Check if this is a numbered list item
            elif re.match(r'^\d+\. ', para_text):
                if not current_list or current_list.style.name != 'List Number':
                    current_list = doc.add_paragraph(style='List Number')
                else:
                    current_list = doc.add_paragraph(style='List Number')
                current_list.text = re.sub(r'^\d+\. ', '', para_text).strip()
                
            # Check if this is a link
            elif '<a href=' in para_text:
                current_list = None  # Reset list context
                p = doc.add_paragraph()
                # Extract link text and URL
                match = re.search(r'<a href="([^"]+)">([^<]+)</a>', para_text)
                if match:
                    url, text = match.groups()
                    run = p.add_run(text)
                    run.hyperlink = url
                else:
                    p.text = para_text
                    
            else:
                current_list = None  # Reset list context
                p = doc.add_paragraph()
                p.text = para_text
        
        # Add logs section
        doc.add_heading('Execution Logs', level=1)
        for log in logs:
            log_para = doc.add_paragraph()
            log_para.text = log
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"job_search_{timestamp}.docx"
        
        # Save the document
        file_path = os.path.join(results_dir, filename)
        doc.save(file_path)
        
        return file_path
        
    except Exception as e:
        raise ValueError(f"Error creating document: {str(e)}")

def format_document(doc: Document) -> Document:
    """
    Apply consistent formatting to a document.
    
    Args:
        doc: The Document object to format.
        
    Returns:
        Document: The formatted Document object.
        
    Raises:
        ValueError: If the document cannot be properly formatted.
    """
    try:
        # Set the default font
        style = doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        
        # Configure heading styles
        for i in range(1, 4):
            heading_style = doc.styles[f'Heading {i}']
            heading_style.font.name = 'Calibri'
            heading_style.font.size = Pt(14 - i)  # Decreasing size for lower levels
            heading_style.font.bold = True
            heading_style.paragraph_format.space_after = Pt(12)
        
        # Configure list styles
        list_bullet = doc.styles['List Bullet']
        list_bullet.font.name = 'Calibri'
        list_bullet.font.size = Pt(11)
        list_bullet.paragraph_format.left_indent = Inches(0.25)
        list_bullet.paragraph_format.space_after = Pt(6)
        
        list_number = doc.styles['List Number']
        list_number.font.name = 'Calibri'
        list_number.font.size = Pt(11)
        list_number.paragraph_format.left_indent = Inches(0.25)
        list_number.paragraph_format.space_after = Pt(6)
        
        # Configure hyperlink style
        try:
            hyperlink_style = doc.styles['Hyperlink']
        except KeyError:
            # Create the hyperlink style if it doesn't exist
            hyperlink_style = doc.styles.add_style('Hyperlink', WD_STYLE_TYPE.CHARACTER)
        
        hyperlink_style.font.color.rgb = RGBColor(0, 0, 255)  # Blue
        hyperlink_style.font.underline = True
        
        # Set the margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)
            
            # Set page size to A4
            section.page_height = Inches(11.69)
            section.page_width = Inches(8.27)
        
        return doc
        
    except Exception as e:
        raise ValueError(f"Error formatting document: {str(e)}")
