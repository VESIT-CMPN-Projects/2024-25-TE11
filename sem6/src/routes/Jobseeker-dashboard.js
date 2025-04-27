import React, { useState, useEffect } from "react";
import { db } from "../firebase-config";
import { collection, getDocs, getDoc, doc, where, query, orderBy, limit, updateDoc, onSnapshot} from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import SearchBar from "../components/SearchBar";
import MyResume from "../components/MyResume";
import AppliedJobsPage from "../components/AppliedJobs";
import ProfilePage from "../components/Profile";
import SkillAnalyzer from "../components/skillAnalyzer";
import { getAuth } from "firebase/auth";
import CourseCompletionCheck from "../components/CourseCompletionCheck";
import JobsListingPage from "../components/JobDisplay";
import ProjectSuggester from "../components/ProjectSuggester";
import AIMockInterview from "../components/AIMockInterview";
import ProjectAssistant from "../components/ProjectAssistant";
import JobNotificationModal from "../components/JobNotificationModal";
import MentorshipPlatform from "../components/Mentorship";
import FetchProblem from "../components/CodingChallenges";
 
const JobseekerDashboard = () => {
  const [activeTab, setActiveTab] = useState("home");
  const [currentUser, setCurrentUser] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredJobs, setFilteredJobs] = useState([]);
  const [userProfile, setUserProfile] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showNotificationAlert, setShowNotificationAlert] = useState(false);
  const [currentAlert, setCurrentAlert] = useState(null);
  const [showJobModal, setShowJobModal] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const navigate = useNavigate();
  const auth = getAuth();
 
  useEffect(() => {
    const fetchUserProfile = async () => {
      if (!auth.currentUser) {
        console.error("User not authenticated");
        setLoading(false);
        return;
      }
 
      try {
        const userRef = doc(db, "userProfiles", auth.currentUser.uid);
        const userSnap = await getDoc(userRef);
 
        if (userSnap.exists()) {
          setUserProfile(userSnap.data());
        } else {
          console.error("User profile not found");
        }
      } catch (error) {
        console.error("Error fetching user profile:", error);
      } finally {
        setLoading(false);
      }
    };
 
    fetchUserProfile();
  }, []);

  useEffect(() => {
    if (!auth.currentUser) return;
    
    const fetchNotifications = async () => {
      try {
        // Get the 20 most recent notifications
        const notificationsRef = collection(db, "notifications");
        const q = query(
          notificationsRef,
          orderBy("createdAt", "desc"),
          limit(20)
        );
        
        // Set up a real-time listener
        const unsubscribe = onSnapshot(q, (snapshot) => {
          const notificationData = snapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data(),
            read: doc.data().readBy && doc.data().readBy.includes(auth.currentUser.uid)
          }));
          
          // Check if there are new notifications (compare with previous state)
          if (notificationData.length > notifications.length) {
            // Find the newest unread notification
            const newestNotification = notificationData[0];
            if (newestNotification && !newestNotification.read) {
              setCurrentAlert(newestNotification);
              setShowNotificationAlert(true);
              
              // Auto-hide after 5 seconds
              setTimeout(() => {
                setShowNotificationAlert(false);
              }, 5000);
            }
          }
          
          setNotifications(notificationData);
          
          // Count unread notifications
          const unread = notificationData.filter(n => !n.read).length;
          setUnreadCount(unread);
        });
        
        return unsubscribe;
      } catch (error) {
        console.error("Error fetching notifications:", error);
        return () => {}; // Return empty function as fallback
      }
    };
    
    // Store the returned unsubscribe function
    const unsubscribe = fetchNotifications();
    
    // Cleanup listener on unmount
    return () => {
      if (typeof unsubscribe === 'function') {
        unsubscribe();
      }
    };
  }, [auth.currentUser, notifications.length]);
  
  // Function to mark notification as read
  const markAllAsRead = async () => {
    if (!auth.currentUser) return;
    
    try {
      // Process each unread notification
      const unreadNotifications = notifications.filter(n => !n.read);
      
      // Use Promise.all to handle all updates in parallel
      await Promise.all(
        unreadNotifications.map(async (notification) => {
          const notificationRef = doc(db, "notifications", notification.id);
          const notificationSnap = await getDoc(notificationRef);
          
          if (notificationSnap.exists()) {
            const data = notificationSnap.data();
            if (!data.readBy || !data.readBy.includes(auth.currentUser.uid)) {
              await updateDoc(notificationRef, {
                readBy: [...(data.readBy || []), auth.currentUser.uid]
              });
            }
          }
        })
      );
    } catch (error) {
      console.error("Error marking all notifications as read:", error);
    }
  };
  
  const handleNotificationClick = (notification) => {
    // Mark as read
    markNotificationAsRead(notification.id);
    
    // If it's a job notification, show the job modal
    if (notification.type === "newJob" && notification.jobId) {
      setSelectedJobId(notification.jobId);
      setShowJobModal(true);
      setShowNotifications(false); // Close the notifications dropdown
    } else {
      // For other notification types, keep existing behavior
      if (notification.type === "newJob") {
        setActiveTab("home");
        setSearchQuery(notification.jobTitle); // This will filter to show the job
      }
      
      setShowNotifications(false);
    }
  };

  const markNotificationAsRead = async (notificationId) => {
    if (!auth.currentUser) return;
    
    try {
      const notificationRef = doc(db, "notifications", notificationId);
      const notificationSnap = await getDoc(notificationRef);
      
      if (notificationSnap.exists()) {
        const data = notificationSnap.data();
        if (!data.readBy || !data.readBy.includes(auth.currentUser.uid)) {
          await updateDoc(notificationRef, {
            readBy: [...(data.readBy || []), auth.currentUser.uid]
          });
        }
      }
    } catch (error) {
      console.error("Error marking notification as read:", error);
    }
  };
 
  // Auth state listener
  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (user) {
        setCurrentUser(user);
      } else {
        // No user is signed in, redirect to login
        navigate('/login');
      }
    });
 
    // Cleanup subscription
    return () => unsubscribe();
  }, [navigate]);
 
  // Fetch jobs from Firestore
  // In JobseekerDashboard.js, modify the useEffect for fetching jobs:
useEffect(() => {
  const fetchJobs = async () => {
    try {
      setLoading(true);
      
      // Just fetch all active jobs without filtering here
      const jobsRef = collection(db, "jobs");
      const q = query(jobsRef, where("status", "==", "active"));
      const querySnapshot = await getDocs(q);
        
      const jobsData = querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
        
      setJobs(jobsData);
      setFilteredJobs(jobsData);  // Let JobsListingPage handle the filtering
      
    } catch (error) {
      console.error("Error fetching jobs:", error);
      setJobs([]);
      setFilteredJobs([]);
    } finally {
      setLoading(false);
    }
  };

  fetchJobs();
}, []);
 
  // Handle search functionality
  useEffect(() => {
    if (searchQuery) {
      const lowercaseQuery = searchQuery.toLowerCase();
      const filtered = jobs.filter(job =>
        job.jobTitle?.toLowerCase().includes(lowercaseQuery) ||
        job.companyName?.toLowerCase().includes(lowercaseQuery) ||
        job.jobDescription?.toLowerCase().includes(lowercaseQuery) ||
        job.category?.toLowerCase().includes(lowercaseQuery) ||
        (job.requiredSkills && job.requiredSkills.some(skill =>
          skill.toLowerCase().includes(lowercaseQuery)
        ))
      );
      setFilteredJobs(filtered);
    } else {
      setFilteredJobs(jobs);
    }
  }, [searchQuery, jobs]);
 
  const handleLogout = async () => {
    try {
      await auth.signOut();
      navigate('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };
 
  const handleSearch = (query) => {
    setSearchQuery(query);
  };
 
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };
 
  const renderContent = () => {
    switch (activeTab) {
      case "home":
        return <MainContent />;
      case "profile":
        return <ProfilePage userId={currentUser?.uid} />;
      case "resume":
        return <MyResume userId={currentUser?.uid} />;
      case "project":
        return <ProjectSuggester />;
      case "projectprogress":
        return <ProjectAssistant />;
      case "jobs":
        return <AppliedJobsPage />
      case "mentors":
        return <MentorshipPlatform/>
      case "mock-interview":
          return <AIMockInterview />;
      case "tests":
          return <FetchProblem/>;
      default:
        return <MainContent />;
    }
  };
  console.log("currentUser", currentUser);
  console.log("API Key:", process.env.REACT_APP_GOOGLE_API_KEY);
 
  const MainContent = () => {
    return (
      <div className="p-6">
        {showNotificationAlert && currentAlert && (
          <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-blue-100 p-4 max-w-md z-50 animate-slide-up">
            <div className="flex items-start">
              <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 flex-shrink-0">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="ml-3 flex-1">
                <div className="flex justify-between items-start">
                  <p className="text-sm font-medium text-gray-900">
                    {currentAlert.type === "newJob" ? `New job posted: ${currentAlert.jobTitle}` : "New notification"}
                  </p>
                  <button 
                    onClick={() => setShowNotificationAlert(false)}
                    className="text-gray-400 hover:text-gray-500"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  {currentAlert.companyName} • {currentAlert.category}
                </p>
                <div className="mt-2 flex justify-between items-center">
                  <p className="text-xs text-gray-400">
                    {new Date(currentAlert.createdAt).toLocaleString()}
                  </p>
                  <button 
                    onClick={() => {
                      handleNotificationClick(currentAlert);
                      setShowNotificationAlert(false);
                    }}
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    View Details
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        <CourseCompletionCheck />
        <SkillAnalyzer userProfile={userProfile}/>
        <SearchBar onSearch={handleSearch} />
        <JobsListingPage jobs={filteredJobs} loading={loading} userProfile={userProfile} />
        <JobNotificationModal 
          isOpen={showJobModal}
          onClose={() => setShowJobModal(false)}
          jobId={selectedJobId}
        />
      </div>
    );
  };
 
  // Header component with fixed icons
  const Header = () => {
    return (
      <header className="bg-white shadow-sm px-6 py-4 flex items-center justify-between">
        <div className="flex items-center">
          <button
            onClick={toggleSidebar}
            className="mr-4 text-gray-500 hover:text-gray-700 focus:outline-none"
          >
            {/* Inline SVG for menu icon instead of Icons.Menu */}
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 className="text-xl font-bold text-blue-600">SkillBridgeAI</h1>
        </div>
       
        <div className="flex items-center">
          <div className="flex flex-row space-x-4">
          <div className="relative">
          <button 
            onClick={() => setShowNotifications(!showNotifications)}
            className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100 relative"
          >
            {/* Bell icon */}
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            
            {/* Notification badge */}
            {unreadCount > 0 && (
              <span className="absolute top-0 right-0 h-5 w-5 text-xs flex items-center justify-center bg-red-500 text-white rounded-full">
                {unreadCount}
              </span>
            )}
          </button>
          
          {/* Notifications dropdown */}
          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-white rounded-md shadow-lg overflow-hidden z-20">
              <div className="py-2">
                <div className="px-4 py-2 bg-gray-50 border-b flex justify-between items-center">
                  <h3 className="font-medium text-gray-700">Notifications</h3>
                  {unreadCount > 0 && (
                    <button 
                      onClick={markAllAsRead}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      Mark all as read
                    </button>
                  )}
                </div>
                
                <div className="max-h-80 overflow-y-auto">
                  {notifications.length === 0 ? (
                    <p className="text-center py-4 text-gray-500">No notifications</p>
                  ) : (
                    notifications.map(notification => (
                      <div 
                        key={notification.id}
                        onClick={() => handleNotificationClick(notification)}
                        className={`px-4 py-3 hover:bg-gray-50 cursor-pointer border-b ${!notification.read ? 'bg-blue-50' : ''}`}
                      >
                        <div className="flex items-start">
                          <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 flex-shrink-0">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <div className="ml-3 flex-1">
                            <p className="text-sm font-medium text-gray-900">
                              {notification.type === "newJob" ? `New job posted: ${notification.jobTitle}` : "Notification"}
                            </p>
                            <p className="text-xs text-gray-500">
                              {notification.companyName} • {notification.category}
                            </p>
                            <p className="text-xs text-gray-400 mt-1">
                              {new Date(notification.createdAt).toLocaleString()}
                            </p>
                          </div>
                          {!notification.read && (
                            <span className="h-2 w-2 bg-blue-600 rounded-full"></span>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
            <button className="flex items-center space-x-2 group">
              <div className="h-9 w-9 rounded-full bg-blue-600 flex items-center justify-center text-white font-medium overflow-hidden">
                {userProfile?.profilePicture ? (
                  <img src={userProfile.profilePicture} alt="Profile" className="h-full w-full object-cover" />
                ) : (
                  userProfile?.firstName?.charAt(0) || currentUser?.email?.charAt(0) || "U"
                )}
              </div>
              <span className="text-sm font-medium text-gray-700 group-hover:text-blue-600">
                {userProfile?.firstName || currentUser?.email?.split('@')[0] || "User"}
              </span>
            </button>
          </div>
        </div>
      </header>
    );
  };
 
  // Sidebar component with fixed icons
  const Sidebar = ({ activeTab, setActiveTab, onLogout, collapsed }) => {
    // Direct SVG icons instead of using the Icons component
    const homeIcon = (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    );
   
    const userIcon = (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    );
   
    const chartIcon = (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    );
   
    const fileIcon = (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
   
    const briefcaseIcon = (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    );
   
    const logoutIcon = (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
      </svg>
    );
 
    const menuItems = [
      { id: "home", label: "Home", icon: homeIcon },
      { id: "profile", label: "Profile", icon: userIcon },
      { id: "resume", label: "My Resume", icon: fileIcon },
      { id: "project", label: "Projects", icon: chartIcon },
      { id: "projectprogress", label: "Project Progress", icon: chartIcon },
      { id: "jobs", label: "Applied Jobs", icon: briefcaseIcon },
      { id: "mentors", label: "Mentorship", icon: userIcon },
      { id: "mock-interview", label: " AI Mock Interview", icon: briefcaseIcon },
      { id: "tests", label: "Coding Tests", icon: chartIcon },
    ];
 
    return (
      <div
        className={`h-screen bg-white shadow-lg fixed left-0 top-0 flex flex-col transition-all duration-300 z-20 ${
          collapsed ? "w-20" : "w-64"
        }`}
      >
        <div className={`p-4 border-b flex items-center ${collapsed ? "justify-center" : "justify-between"}`}>
          {!collapsed && <h1 className="text-xl font-bold text-blue-600">Dashboard</h1>}
          {collapsed && (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          )}
        </div>
       
        <nav className="mt-6 flex-grow">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center px-4 py-4 hover:bg-blue-50 transition-colors ${
                activeTab === item.id
                  ? "bg-blue-50 text-blue-600 border-r-4 border-blue-600"
                  : "text-gray-700"
              } ${collapsed ? "justify-center" : ""}`}
            >
              <span className={`${collapsed ? "text-center" : ""}`}>{item.icon}</span>
              {!collapsed && <span className="ml-3 font-medium">{item.label}</span>}
            </button>
          ))}
        </nav>
       
        <div className="p-4 border-t">
          <button
            onClick={onLogout}
            className={`w-full flex items-center px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors ${
              collapsed ? "justify-center" : ""
            }`}
          >
            {logoutIcon}
            {!collapsed && <span className="ml-3 font-medium">Logout</span>}
          </button>
        </div>
      </div>
    );
  };
 
  return (
    <div className="bg-gray-50 min-h-screen">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onLogout={handleLogout}
        collapsed={sidebarCollapsed}
      />
     
      <div className={`transition-all duration-300 ${
        sidebarCollapsed ? "ml-20" : "ml-64"
      }`}>
        <Header />
        <main className="container mx-auto py-6 px-6">
          {currentUser ? renderContent() : (
            <div className="flex justify-center items-center h-screen">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading your dashboard...</p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
 
export default JobseekerDashboard;
