// NotificationBell.jsx - Add this to your user dashboard
import React, { useState, useEffect } from "react";
import { collection, query, where, onSnapshot, orderBy, updateDoc, doc } from "firebase/firestore";
import { db, auth } from "../firebase-config";
import { useNavigate } from "react-router-dom";

const NotificationBell = () => {
  const [notifications, setNotifications] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const navigate = useNavigate();
  
  useEffect(() => {
    const currentUser = auth.currentUser;
    
    if (!currentUser) return;
    
    const notificationsQuery = query(
      collection(db, "notifications"),
      where("userId", "==", currentUser.uid),
      orderBy("createdAt", "desc")
    );
    
    const unsubscribe = onSnapshot(notificationsQuery, (snapshot) => {
      const notificationsList = [];
      snapshot.forEach((doc) => {
        notificationsList.push({
          id: doc.id,
          ...doc.data()
        });
      });
      
      setNotifications(notificationsList);
      setUnreadCount(notificationsList.filter(n => !n.read).length);
    });
    
    return () => unsubscribe();
  }, []);
  
  const markAsRead = async (notificationId) => {
    const notificationRef = doc(db, "notifications", notificationId);
    await updateDoc(notificationRef, {
      read: true
    });
  };
  
  // In NotificationBell.jsx
    const handleJoinMeeting = async (notification) => {
    await markAsRead(notification.id);
    navigate(`/meeting-room/${notification.meetingId}`);
  };
  
  return (
    <div className="relative">
      <button 
        onClick={() => setShowDropdown(!showDropdown)}
        className="relative p-1 rounded-full text-gray-400 hover:text-indigo-600 focus:outline-none"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
        </svg>
        
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 block h-4 w-4 rounded-full bg-red-500 text-white text-xs font-bold flex items-center justify-center">
            {unreadCount}
          </span>
        )}
      </button>
      
      {showDropdown && (
        <div className="origin-top-right absolute right-0 mt-2 w-80 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
          <div className="py-1 max-h-96 overflow-y-auto" role="menu">
            <div className="px-4 py-2 border-b border-gray-100">
              <h3 className="text-sm font-medium text-gray-900">Notifications</h3>
            </div>
            
            {notifications.length === 0 ? (
              <div className="px-4 py-3 text-sm text-gray-500">
                No notifications
              </div>
            ) : (
              notifications.map((notification) => (
                <div 
                  key={notification.id} 
                  className={`px-4 py-3 hover:bg-gray-50 ${!notification.read ? "bg-indigo-50" : ""}`}
                >
                  <div className="flex justify-between">
                    <p className="text-sm font-medium text-gray-900">{notification.title}</p>
                    <p className="text-xs text-gray-500">
                      {new Date(notification.createdAt).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                    </p>
                  </div>
                  
                  <p className="text-sm text-gray-500 mt-1">{notification.message}</p>
                  
                  {(notification.type === "meeting-started" || notification.type === "session-scheduled") && (
                        <button
                        onClick={() => handleJoinMeeting(notification)}
                        className="mt-2 w-full py-1 px-3 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded transition duration-300"
                        >
                        {notification.type === "meeting-started" ? "Join Meeting" : "Start Meeting"}
                        </button>
                    )}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationBell;