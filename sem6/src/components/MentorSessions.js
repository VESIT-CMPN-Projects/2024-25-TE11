import React, { useState, useEffect } from "react";
import { format, parseISO, isAfter, isBefore, isSameDay } from "date-fns";
import { db, auth } from "../firebase-config";
import { collection, getDocs, query, where, doc, updateDoc,addDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";

const MentorSessions = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("upcoming");
  const [userDetails, setUserDetails] = useState({});
  const navigate = useNavigate();
  
  useEffect(() => {
    const fetchMentorSessions = async () => {
      try {
        setLoading(true);
        const currentUser = auth.currentUser;
        
        if (currentUser) {
          const mentorId = currentUser.uid;
          // Query sessions where mentorId matches the current user's ID
          const sessionsCollection = collection(db, "sessions");
          const q = query(sessionsCollection, where("mentorId", "==", mentorId));
          const querySnapshot = await getDocs(q);
          
          const sessionsList = [];
          querySnapshot.forEach((doc) => {
            sessionsList.push({
              id: doc.id,
              ...doc.data()
            });
          });
          
          // Fetch user details for each session
          const userIds = [...new Set(sessionsList.map(session => session.userId))];
          const userMap = {};
          
          for (const userId of userIds) {
            // In a real app, you would query the users collection
            // For this example, we'll simulate user data
            userMap[userId] = await fetchUserDetails(userId);
          }
          
          setUserDetails(userMap);
          setSessions(sessionsList);
        } else {
          // Mock data for testing when user is not authenticated
          const mockSessions = [
            {
              id: "s1",
              userId: "B8r8FsGnQWdoucRDpEYW6QawwK13",
              mentorId: "pDi2zQb3FyTJ4EgaxYgQ2gwyRYG2",
              date: "2025-03-25T10:00:00",
              status: "upcoming",
              topic: "New Session",
              notes: "",
              hourlyRate: 125,
              createdAt: "2025-03-22T23:21:01.846Z"
            },
            {
              id: "s2",
              userId: "user123",
              mentorId: "pDi2zQb3FyTJ4EgaxYgQ2gwyRYG2",
              date: "2025-03-26T14:00:00",
              status: "upcoming",
              topic: "JavaScript Performance Optimization",
              notes: "",
              hourlyRate: 125,
              createdAt: "2025-03-20T19:45:30.123Z"
            },
            {
              id: "s3",
              userId: "user456",
              mentorId: "pDi2zQb3FyTJ4EgaxYgQ2gwyRYG2",
              date: "2025-03-18T11:00:00",
              status: "completed",
              topic: "React Hooks Deep Dive",
              notes: "Covered useEffect, useMemo, and custom hooks. Student showed good understanding.",
              hourlyRate: 125,
              createdAt: "2025-03-15T14:30:10.789Z"
            }
          ];
          
          // Mock user details
          const mockUsers = {
            "B8r8FsGnQWdoucRDpEYW6QawwK13": {
              name: "Sushanth Shetty",
              email: "sushant@gmail.com",
              photoURL: null
            },
            "user123": {
              name: "Emily Johnson",
              email: "emily.j@example.com",
              photoURL: null
            },
            "user456": {
              name: "Michael Chen",
              email: "michael.chen@example.com",
              photoURL: null
            }
          };
          
          setUserDetails(mockUsers);
          setSessions(mockSessions);
        }
        
        setLoading(false);
      } catch (error) {
        console.error("Error fetching mentor sessions:", error);
        setLoading(false);
      }
    };
    
    fetchMentorSessions();
  }, []);
  
  // Simulate fetching user details
  // In a real app, this would be a query to your users collection
  const fetchUserDetails = async (userId) => {
    // For demo purposes, return mock data
    return {
      name: "Sushanth Shetty",
      email: "sushanth@gmail.com",
      photoURL: null
    };
  };
  
  // MentorSessions.jsx - Update the handleStartSession function
const handleStartSession = async (sessionId) => {
    try {
      const session = sessions.find(s => s.id === sessionId);
      
      // Create a meeting ID (you can use a more robust method in production)
      const meetingId = `meet-${sessionId}-${Date.now()}`;
      
      // Update the session in Firestore
      const sessionRef = doc(db, "sessions", sessionId);
      await updateDoc(sessionRef, {
        status: "in-progress",
        meetingId: meetingId,
        startedAt: new Date().toISOString()
      });
      
      // Send notifications to both mentor and user
      await sendMeetingNotification(session.userId, meetingId, session);
      
      // Redirect mentor to the meeting room
      navigate(`/meeting-room/${meetingId}`);
      
    } catch (error) {
      console.error("Error starting session:", error);
      alert("There was an error starting the session. Please try again.");
    }
  };
  
  // Notification function
  const sendMeetingNotification = async (userId, meetingId, sessionData) => {
    try {
      // Create notification for the user
      const userNotificationRef = collection(db, "notifications");
      await addDoc(userNotificationRef, {
        userId: userId,
        type: "meeting-started",
        title: "Your mentoring session is starting",
        message: `Your mentor has started the session about "${sessionData.topic}"`,
        meetingId: meetingId,
        sessionId: sessionData.id,
        createdAt: new Date().toISOString(),
        read: false
      });
      
      // You might also want to send an email or push notification here
      
    } catch (error) {
      console.error("Error sending notification:", error);
    }
  };
  
  const filteredSessions = sessions.filter(session => {
    const sessionDate = parseISO(session.date);
    const now = new Date();
    
    if (activeTab === "upcoming") {
      return session.status === "upcoming" && isAfter(sessionDate, now);
    } else if (activeTab === "completed") {
      return session.status === "completed" || isBefore(sessionDate, now);
    }
    return true;
  });
  
  const sortedSessions = [...filteredSessions].sort((a, b) => {
    return parseISO(a.date) - parseISO(b.date);
  });
  
  const getInitials = (name) => {
    if (!name) return "?";
    return name.split(" ").map(n => n[0]).join("").toUpperCase();
  };
  
  const renderSessionCard = (session) => {
    const user = userDetails[session.userId] || { name: "Unknown User", email: "", photoURL: null };
    const sessionDate = parseISO(session.date);
    const isUpcoming = isAfter(sessionDate, new Date());
    
    return (
      <div key={session.id} className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
        <div className="p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 font-bold text-lg overflow-hidden">
                {user.photoURL ? (
                  <img src={user.photoURL} alt={user.name} className="w-full h-full object-cover" />
                ) : (
                  getInitials(user.name)
                )}
              </div>
              <div>
                <h4 className="font-medium text-gray-900">{user.name}</h4>
                <p className="text-sm text-gray-500">{user.email}</p>
              </div>
            </div>
            <div className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${
              isUpcoming 
                ? "bg-green-100 text-green-800" 
                : "bg-gray-100 text-gray-800"
            }`}>
              {isUpcoming ? "Upcoming" : "Completed"}
            </div>
          </div>
          
          <div className="mt-4 space-y-2">
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              {format(sessionDate, "MMMM d, yyyy")}
            </div>
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {format(sessionDate, "h:mm a")} (60 minutes)
            </div>
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
              {session.topic}
            </div>
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              ${session.hourlyRate}
            </div>
          </div>
          
          {session.notes && (
            <div className="mt-4 p-3 bg-gray-50 rounded-md text-sm">
              <span className="font-medium">Notes:</span> {session.notes}
            </div>
          )}
          
          <div className="mt-4">
            {isUpcoming ? (
              <button
                onClick={() => handleStartSession(session.id)}
                className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition duration-300"
              >
                Start Session
              </button>
            ) : (
              <div className="flex space-x-3">
                <button className="flex-1 py-2 px-4 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium rounded-lg transition duration-300">
                  Session Notes
                </button>
                <button className="flex-1 py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition duration-300">
                  Follow Up
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">My Mentoring Sessions</h1>
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab("upcoming")}
            className={`py-2 px-4 text-sm font-medium rounded-lg ${
              activeTab === "upcoming"
                ? "bg-indigo-600 text-white"
                : "bg-white border border-gray-300 text-gray-700 hover:bg-gray-50"
            }`}
          >
            Upcoming
          </button>
          <button
            onClick={() => setActiveTab("completed")}
            className={`py-2 px-4 text-sm font-medium rounded-lg ${
              activeTab === "completed"
                ? "bg-indigo-600 text-white"
                : "bg-white border border-gray-300 text-gray-700 hover:bg-gray-50"
            }`}
          >
            Completed
          </button>
        </div>
      </div>
      
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Array(3).fill(0).map((_, idx) => (
            <div key={idx} className="bg-white p-5 rounded-lg shadow animate-pulse">
              <div className="flex items-center space-x-4">
                <div className="rounded-full bg-gray-300 h-12 w-12"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-300 rounded w-5/6"></div>
                </div>
              </div>
              <div className="mt-4 space-y-2">
                <div className="h-3 bg-gray-300 rounded"></div>
                <div className="h-3 bg-gray-300 rounded"></div>
                <div className="h-3 bg-gray-300 rounded"></div>
                <div className="h-3 bg-gray-300 rounded"></div>
              </div>
              <div className="mt-4 h-8 bg-gray-300 rounded"></div>
            </div>
          ))}
        </div>
      ) : sortedSessions.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {sortedSessions.map(session => renderSessionCard(session))}
        </div>
      ) : (
        <div className="text-center py-16 bg-white rounded-lg shadow-md">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="mt-2 text-lg font-medium text-gray-900">No {activeTab} sessions found</h3>
          <p className="mt-1 text-gray-500">
            {activeTab === "upcoming" 
              ? "You don't have any upcoming mentoring sessions."
              : "You haven't completed any mentoring sessions yet."}
          </p>
        </div>
      )}
    </div>
  );
};

export default MentorSessions;