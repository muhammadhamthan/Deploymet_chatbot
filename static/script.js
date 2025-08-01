const chatbotBtn = document.getElementById('chatbot-btn');
const chatPopup = document.getElementById('chat-popup');
const closeBtn = document.getElementById('close-btn');
const sendBtn = document.getElementById('send-btn');
const chatInput = document.getElementById('chat-input');
const chatBody = document.getElementById('chat-body');
const refreshBtn = document.getElementById('refresh-btn');

// Hide chatbot button if chat popup is visible on load
if (chatPopup.style.display !== 'flex') {
  chatbotBtn.style.display = 'none';
}

// Prevent scroll to top on button click
// Open chatbot and hide open button
chatbotBtn.addEventListener('click', (e) => {
  e.preventDefault();
  chatPopup.style.display = 'flex';
  chatbotBtn.style.display = 'none';
});

// Close chatbot and show open button
closeBtn.addEventListener('click', () => {
  chatPopup.style.display = 'none';
  chatbotBtn.style.display = 'flex';
});


// Refresh chat
refreshBtn.addEventListener('click', () => {
  chatBody.innerHTML =
    '<div class="bot-message-wrapper">' +
      '<div class="bot-avatar"></div>' +
      '<div class="bot-message">Hi! How can I help you today?</div>' +
    '</div>';
});


// Send message
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
});

function sendMessage() {
  const userMessage = chatInput.value.trim();
  if (userMessage !== '') {
    displayMessage(userMessage, 'user-message');

    const typingBubble = document.createElement('div');
    typingBubble.classList.add('bot-message');
    typingBubble.id = 'typing-indicator';
    typingBubble.innerHTML = '<em>Typing<span class="dots"></span></em>';
    chatBody.appendChild(typingBubble);
    chatBody.scrollTop = chatBody.scrollHeight;

    fetch('http://127.0.0.1:50001/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage })
    })
    .then(response => response.json())
    .then(data => {
      document.getElementById('typing-indicator')?.remove();
      if (data && data.response) {
        displayBotResponse(data.response);
      } else {
        displayMessage("Sorry, something went wrong with the response format.", 'bot-message', true);
      }
    })
    .catch(error => {
      document.getElementById('typing-indicator')?.remove();
      console.error('Error:', error);
      displayMessage("Sorry, I couldn't process your request.", 'bot-message', true);
    });

    chatInput.value = '';
    chatBody.scrollTop = chatBody.scrollHeight;
  }
}

function displayMessage(message, type, isBot = false) {
  const messageElement = document.createElement('div');
  messageElement.classList.add(type);
  messageElement.innerHTML = isBot ? message : escapeHtml(message);
  chatBody.appendChild(messageElement);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function displayBotResponse(response) {
  if (response) {
    const points = response.split('\n');

    const wrapper = document.createElement('div');
    wrapper.classList.add('bot-message-wrapper');

    const avatar = document.createElement('div');
    avatar.classList.add('bot-avatar');

    const botResponseContainer = document.createElement('div');
    botResponseContainer.classList.add('bot-message');

    points.forEach(point => {
      if (point.trim() !== '') {
        const pointElement = document.createElement('div');
        const formattedText = point.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        pointElement.innerHTML = '<span class="point">– </span>' + formattedText;
        botResponseContainer.appendChild(pointElement);
      }
    });

    wrapper.appendChild(avatar);
    wrapper.appendChild(botResponseContainer);
    chatBody.appendChild(wrapper);
    chatBody.scrollTop = chatBody.scrollHeight;
  } else {
    displayMessage("Sorry, I couldn't get a valid response.", 'bot-message', true);
  }
}


function escapeHtml(text) {
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}
