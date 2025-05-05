document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-prompts-input');
    const newPromptText = document.getElementById('new-prompt-text');
    const saveNewPromptButton = document.getElementById('save-new-prompt-button');
    
    // Load prompts initially
    loadAndDisplayPrompts();

    // Add listener for search input
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            loadAndDisplayPrompts(searchInput.value);
        });
    }

    // Add listener for saving new prompt
    if (saveNewPromptButton && newPromptText) {
        saveNewPromptButton.addEventListener('click', async () => {
            const textToSave = newPromptText.value.trim();
            if (!textToSave) {
                showToast('Cannot save an empty prompt', 'warning');
                return;
            }
            try {
                const response = await fetch('/api/prompts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textToSave })
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
                }
                showToast('Prompt saved successfully');
                newPromptText.value = ''; // Clear input after saving
                loadAndDisplayPrompts(); // Refresh the list (clears search if any)
            } catch (error) {
                console.error('Error saving prompt:', error);
                showToast(`Failed to save prompt: ${error.message}`, 'error');
            }
        });
    }
});

// --- Utility Functions (Consider moving to a shared utils.js if needed elsewhere) ---
function showToast(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // Basic toast for now, replace with visual element if desired
    // Example: Add temporary message near the top
    const toastContainer = document.body;
    const toastElement = document.createElement('div');
    toastElement.className = `alert alert-${type} position-fixed top-0 end-0 m-3 fade show`; 
    toastElement.style.zIndex = 1050;
    toastElement.textContent = message;
    toastContainer.appendChild(toastElement);
    setTimeout(() => {
        toastElement.classList.remove('show');
        setTimeout(() => toastElement.remove(), 150);
    }, 3000);
}

// --- Prompt Management Functions ---
async function loadAndDisplayPrompts(query = '') {
    const listElement = document.getElementById('saved-prompts-list');
    if (!listElement) return;
    listElement.innerHTML = '<div class="list-group-item text-muted small">Loading...</div>'; 

    try {
        const url = query ? `/api/prompts/search?query=${encodeURIComponent(query)}` : '/api/prompts';
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const prompts = await response.json();
        displayPrompts(prompts);
    } catch (error) {
        console.error('Error loading prompts:', error);
        listElement.innerHTML = '<div class="list-group-item text-danger small">Error loading prompts.</div>';
        showToast('Failed to load saved prompts', 'error');
    }
}

function displayPrompts(prompts) {
    const listElement = document.getElementById('saved-prompts-list');
    if (!listElement) return;

    listElement.innerHTML = ''; // Clear current list

    if (prompts.length === 0) {
        listElement.innerHTML = '<div class="list-group-item text-muted small">No prompts saved yet.</div>';
        return;
    }

    prompts.forEach(prompt => {
        const item = document.createElement('div');
        item.className = 'list-group-item d-flex justify-content-between align-items-start py-2 px-3'; // Changed align-items-center to align-items-start

        // Container for text and show more button
        const textContainer = document.createElement('div');
        textContainer.className = 'prompt-text-container me-3'; // Add margin to separate from buttons
        textContainer.style.flexGrow = '1'; // Allow text container to grow

        const textSpan = document.createElement('div'); // Use div for better block handling
        textSpan.textContent = prompt.text;
        textSpan.title = 'Click to expand/collapse'; // Add title for hover hint
        textSpan.className = 'prompt-text'; // Add class for styling

        const showMoreBtn = document.createElement('button');
        showMoreBtn.className = 'btn btn-link btn-sm p-0 show-more-btn'; // Use btn-link style
        showMoreBtn.textContent = 'Show More';

        // Toggle expansion on click
        showMoreBtn.onclick = (e) => {
            e.stopPropagation(); // Prevent triggering item click if any
            textContainer.classList.toggle('expanded');
            if (textContainer.classList.contains('expanded')) {
                showMoreBtn.textContent = 'Show Less';
                textSpan.title = 'Click to collapse';
            } else {
                showMoreBtn.textContent = 'Show More';
                textSpan.title = 'Click to expand';
            }
        };
        // Also allow toggling by clicking the text itself
        textSpan.onclick = () => showMoreBtn.click();

        textContainer.appendChild(textSpan);
        // Only add "Show More" button if the text is likely to overflow
        // We'll refine this logic after adding CSS, for now, always add it for testing
        // A simple check (e.g., text length) could be added here later.
        textContainer.appendChild(showMoreBtn);


        const buttonGroup = document.createElement('div');
        buttonGroup.className = 'btn-group btn-group-sm flex-shrink-0';

        const loadButton = document.createElement('button');
        loadButton.className = 'btn btn-outline-primary';
        loadButton.innerHTML = '<i class="fas fa-arrow-up"></i> Use Prompt';
        loadButton.title = 'Use this prompt on the Home page';
        loadButton.onclick = () => {
            localStorage.setItem('promptToLoad', prompt.text);
            window.location.href = '/';
        };

        const deleteButton = document.createElement('button');
        deleteButton.className = 'btn btn-outline-danger';
        deleteButton.innerHTML = '<i class="fas fa-trash-alt"></i> Delete';
        deleteButton.title = 'Delete this prompt';
        deleteButton.onclick = async () => {
            if (confirm('Are you sure you want to delete this prompt?')) {
                try {
                    const response = await fetch(`/api/prompts/${prompt.id}`, { method: 'DELETE' });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    showToast('Prompt deleted');
                    loadAndDisplayPrompts(document.getElementById('search-prompts-input').value);
                } catch (error) {
                    console.error('Error deleting prompt:', error);
                    showToast('Failed to delete prompt', 'error');
                }
            }
        };

        buttonGroup.appendChild(loadButton);
        buttonGroup.appendChild(deleteButton);

        item.appendChild(textContainer); // Add the text container
        item.appendChild(buttonGroup);
        listElement.appendChild(item);

        // After appending, check if the button should be visible
        // Check if scrollHeight is greater than clientHeight
        if (textSpan.scrollHeight <= textSpan.clientHeight) {
            showMoreBtn.style.display = 'none'; // Hide button if no overflow
        } else {
             textSpan.title = 'Click to expand'; // Initial title if expandable
        }
    });
} 