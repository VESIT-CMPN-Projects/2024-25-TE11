// ChatbotIcon.js
import React, { useState } from 'react';
import { FaCommentDots } from 'react-icons/fa';

function ChatbotIcon() {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);

  const toggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <div style={{ position: 'fixed', bottom: '20px', right: '20px' }}>
      <button
        onClick={toggleChatbot}
        style={{
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          cursor: 'pointer',
          display: 'flex', // Added to center the icon
          justifyContent: 'center', // Added to center the icon
          alignItems: 'center', // Added to center the icon
        }}
      >
        <FaCommentDots size={30} /> {/* Use the icon */}
      </button>
      {isChatbotOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '80px',
            right: '20px',
            width: '400px',
            height: '500px',
            border: '1px solid #ccc',
            backgroundColor: 'white',
            zIndex: 1000,
          }}
        >
          <iframe
            src="https://www.chatbase.co/chatbot-iframe/HIDeqwIgT4JbFh-f7YF7U"
            width="100%"
            height="100%"
            title="Chatbot"
          />
        </div>
      )}
    </div>
  );
}

export default ChatbotIcon;