document.addEventListener('DOMContentLoaded', function() {
    let content = document.querySelector('.content');
    if (content) {
        content.classList.add('fade-in');
    }
});
document.getElementById('story-form').addEventListener('submit', function(event) {
    document.getElementById('loader').style.display = 'block';

    // Clear previous error messages
    const errorMessages = document.getElementById('error-messages');
    if (errorMessages) {
        errorMessages.remove();
    }
});

