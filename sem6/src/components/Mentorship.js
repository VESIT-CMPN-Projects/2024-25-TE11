import React, { useState, useEffect } from "react";
import { format, parseISO, isAfter, isBefore, isSameDay } from "date-fns";
import { db, auth } from "../firebase-config"; // Import Firebase config
import { collection, addDoc, getDocs, query, where } from "firebase/firestore";
import mentorsData from "../data/mentors.csv"; // Using an existing CSV file
import { useNavigate } from 'react-router-dom';

const MentorshipPlatform = () => {
  const [mentors, setMentors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("findMentor");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDomain, setSelectedDomain] = useState("All");
  const [bookedSessions, setBookedSessions] = useState([]);
  const [selectedMentor, setSelectedMentor] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTimeSlot, setSelectedTimeSlot] = useState(null);
  const [bookingStep, setBookingStep] = useState(0);
  const [showBookingModal, setShowBookingModal] = useState(false);

  // Domains for filtering
  const domains = [
    "All",
    "Software Engineering",
    "Full Stack Development",
    "AI & ML Engineering",
    "Cloud and DevOps",
    "Career Guidance",
    "Resume Review",
    "Interview Prep",
  ];

  // Mock fetch mentors from CSV
  useEffect(() => {
    const fetchMentors = async () => {
      try {
        // In a real implementation, you would parse the CSV file
        // Here we're simulating that with mock data
        setTimeout(() => {
          const mockMentors = [
            {
              id: "pDi2zQb3FyTJ4EgaxYgQ2gwyRYG2", // Using the specified mentor ID
              name: "Arjun Sharma",
              title: "Senior Software Engineer",
              company: "Google",
              profilePicture: null,
              rating: "4.8",
              sessionsCompleted: 142,
              expertise: ["JavaScript", "React", "System Design"],
              domains: ["Software Engineering", "Full Stack Development"],
              bio: "10+ years building scalable web applications and mentoring junior developers.",
              availability: [
                { date: "2025-03-24", slots: ["09:00", "10:00", "14:00"] },
                { date: "2025-03-25", slots: ["11:00", "13:00", "16:00"] },
                { date: "2025-03-26", slots: ["09:00", "15:00", "17:00"] },
              ],
              hourlyRate: 120,
            },
            {
              id: "2",
              name: "Sarah Chen",
              title: "AI Research Scientist",
              company: "DeepMind",
              profilePicture: null,
              rating: "4.9",
              sessionsCompleted: 89,
              expertise: ["Machine Learning", "Python", "Neural Networks"],
              domains: ["AI & ML Engineering", "Career Guidance"],
              bio: "PhD in AI with experience in implementing cutting-edge algorithms.",
              availability: [
                { date: "2025-03-23", slots: ["10:00", "13:00", "16:00"] },
                { date: "2025-03-24", slots: ["09:00", "14:00"] },
                { date: "2025-03-27", slots: ["11:00", "15:00", "17:00"] },
              ],
              hourlyRate: 150,
            },
            {
              id: "3",
              name: "Vikram Malhotra",
              title: "DevOps Lead",
              company: "AWS",
              profilePicture: null,
              rating: "4.7",
              sessionsCompleted: 63,
              expertise: ["Kubernetes", "Docker", "CI/CD", "Cloud Architecture"],
              domains: ["Cloud and DevOps", "Software Engineering"],
              bio: "Passionate about automating everything and building resilient systems.",
              availability: [
                { date: "2025-03-23", slots: ["09:00", "11:00", "15:00"] },
                { date: "2025-03-26", slots: ["10:00", "14:00", "16:00"] },
                { date: "2025-03-27", slots: ["09:00", "13:00", "17:00"] },
              ],
              hourlyRate: 135,
            },
            {
              id: "4",
              name: "Priya Patel",
              title: "Technical Recruiter",
              company: "Microsoft",
              profilePicture: null,
              rating: "4.9",
              sessionsCompleted: 217,
              expertise: ["Resume Building", "Interview Coaching", "Career Planning"],
              domains: ["Resume Review", "Interview Prep", "Career Guidance"],
              bio: "Helped 200+ professionals land their dream tech roles.",
              availability: [
                { date: "2025-03-24", slots: ["10:00", "12:00", "15:00"] },
                { date: "2025-03-25", slots: ["09:00", "13:00", "16:00"] },
                { date: "2025-03-27", slots: ["11:00", "14:00", "17:00"] },
              ],
              hourlyRate: 110,
            },
            {
              id: "5",
              name: "Rahul Mehta",
              title: "Full Stack Developer",
              company: "Stripe",
              profilePicture: null,
              rating: "4.6",
              sessionsCompleted: 78,
              expertise: ["TypeScript", "Node.js", "React", "GraphQL"],
              domains: ["Full Stack Development", "Software Engineering"],
              bio: "Building modern web applications with a focus on clean code and best practices.",
              availability: [
                { date: "2025-03-23", slots: ["09:00", "14:00", "16:00"] },
                { date: "2025-03-25", slots: ["10:00", "13:00", "17:00"] },
                { date: "2025-03-26", slots: ["11:00", "15:00"] },
              ],
              hourlyRate: 125,
            },
            {
              id: "6",
              name: "Anjali Desai",
              title: "Product Manager",
              company: "Airbnb",
              profilePicture: null,
              rating: "4.8",
              sessionsCompleted: 104,
              expertise: ["Product Strategy", "Agile", "User Research", "Roadmapping"],
              domains: ["Career Guidance", "Interview Prep"],
              bio: "Experienced in building products used by millions of users worldwide.",
              availability: [
                { date: "2025-03-24", slots: ["11:00", "13:00", "15:00"] },
                { date: "2025-03-26", slots: ["09:00", "12:00", "16:00"] },
                { date: "2025-03-27", slots: ["10:00", "14:00", "17:00"] },
              ],
              hourlyRate: 140,
            },
          ];
          
          setMentors(mockMentors);
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error("Error fetching mentors:", error);
        setLoading(false);
      }
    };

    fetchMentors();
  }, []);

  // Fetch booked sessions from Firebase
  useEffect(() => {
    const fetchBookedSessions = async () => {
      try {
        setLoading(true);
        const currentUser = auth.currentUser;
        
        if (currentUser) {
          const userId = currentUser.uid;
          const sessionsCollection = collection(db, "sessions");
          const q = query(sessionsCollection, where("userId", "==", userId));
          const querySnapshot = await getDocs(q);
          
          const sessions = [];
          querySnapshot.forEach((doc) => {
            sessions.push({
              id: doc.id,
              ...doc.data()
            });
          });
          
          setBookedSessions(sessions);
        } else {
          // Mock data for testing when user is not authenticated
          setTimeout(() => {
            const mockSessions = [
              {
                id: "s1",
                mentorId: "pDi2zQb3FyTJ4EgaxYgQ2gwyRYG2",
                date: "2025-03-18T14:00:00",
                status: "completed",
                topic: "React Performance Optimization",
                notes: "Discussed code splitting and memo techniques."
              },
              {
                id: "s2",
                mentorId: "4",
                date: "2025-03-28T11:00:00",
                status: "upcoming",
                topic: "Resume Review for Senior Developer Roles",
                notes: ""
              },
              {
                id: "s3",
                mentorId: "2",
                date: "2025-03-30T15:00:00",
                status: "upcoming",
                topic: "Introduction to Neural Networks",
                notes: ""
              }
            ];
            
            setBookedSessions(mockSessions);
          }, 1000);
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching booked sessions:", error);
        setLoading(false);
      }
    };

    fetchBookedSessions();
  }, []);

  // Filter mentors based on search and domain
  const filteredMentors = mentors.filter(mentor => {
    const matchesSearch = searchQuery === "" || 
      mentor.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      mentor.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      mentor.company?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      mentor.expertise?.some(skill => skill.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesDomain = selectedDomain === "All" || 
      mentor.domains?.includes(selectedDomain);
    
    return matchesSearch && matchesDomain;
  });

  const handleSearch = (e) => {
    setSearchQuery(e.target.value);
  };

  const handleBookSession = (mentor) => {
    setSelectedMentor(mentor);
    setSelectedDate(null);
    setSelectedTimeSlot(null);
    setBookingStep(0);
    setShowBookingModal(true);
  };

  const handleDateSelect = (date) => {
    setSelectedDate(date);
    setSelectedTimeSlot(null);
    setBookingStep(1);
  };

  const handleTimeSelect = (time) => {
    setSelectedTimeSlot(time);
    setBookingStep(2);
  };

  const handleConfirmBooking = async () => {
    try {
      const currentUser = auth.currentUser;
      const userId = currentUser ? currentUser.uid : "simulated-user-id";
      
      // Create session object
      const newSession = {
        userId: userId,
        mentorId: "pDi2zQb3FyTJ4EgaxYgQ2gwyRYG2",
        date: `${selectedDate}T${selectedTimeSlot}:00`,
        status: "upcoming",
        topic: "New Session",
        notes: "",
        createdAt: new Date().toISOString(),
        hourlyRate: selectedMentor.hourlyRate
      };
      
      // Add to Firebase
      const sessionsCollection = collection(db, "sessions");
      const docRef = await addDoc(sessionsCollection, newSession);
      
      // Update local state with firebase ID
      const sessionWithId = {
        id: docRef.id,
        ...newSession
      };
      
      setBookedSessions([...bookedSessions, sessionWithId]);
      setShowBookingModal(false);
      
      // Show success notification
      alert("Session booked successfully!");
    } catch (error) {
      console.error("Error booking session:", error);
      alert("There was an error booking your session. Please try again.");
    }
  };

  const renderBookingModal = () => {
    if (!showBookingModal) return null;
    
    const today = new Date();
    const availableDates = selectedMentor?.availability || [];
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6 max-h-[90vh] overflow-y-auto">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold text-gray-900">Book a Session</h3>
            <button 
              onClick={() => setShowBookingModal(false)}
              className="text-gray-400 hover:text-gray-500"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          {bookingStep === 0 && (
            <div className="space-y-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-lg overflow-hidden">
                  {selectedMentor.name?.charAt(0) || "M"}
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">{selectedMentor.name}</h4>
                  <p className="text-sm text-gray-500">{selectedMentor.title} at {selectedMentor.company}</p>
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Select a Date</h4>
                <div className="grid grid-cols-3 gap-2">
                  {availableDates.map((dateObj) => (
                    <button
                      key={dateObj.date}
                      onClick={() => handleDateSelect(dateObj.date)}
                      className="p-3 text-center border rounded-lg hover:bg-blue-50 transition-colors"
                    >
                      {format(parseISO(dateObj.date), "MMM d")}
                      <div className="text-xs text-gray-500 mt-1">
                        {dateObj.slots.length} slots
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
          
          {bookingStep === 1 && (
            <div className="space-y-6">
              <button 
                onClick={() => setBookingStep(0)} 
                className="flex items-center text-blue-600 text-sm mb-4"
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to dates
              </button>
              
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">
                  Select a Time Slot for {format(parseISO(selectedDate), "MMMM d, yyyy")}
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {availableDates.find(d => d.date === selectedDate)?.slots.map((time) => (
                    <button
                      key={time}
                      onClick={() => handleTimeSelect(time)}
                      className="p-3 text-center border rounded-lg hover:bg-blue-50 transition-colors"
                    >
                      {time}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
          
          {bookingStep === 2 && (
            <div className="space-y-6">
              <button 
                onClick={() => setBookingStep(1)} 
                className="flex items-center text-blue-600 text-sm mb-4"
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to time slots
              </button>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-900 mb-4">Booking Summary</h4>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Mentor:</span>
                    <span className="font-medium">{selectedMentor.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Date:</span>
                    <span className="font-medium">{format(parseISO(selectedDate), "MMMM d, yyyy")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Time:</span>
                    <span className="font-medium">{selectedTimeSlot}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Duration:</span>
                    <span className="font-medium">60 minutes</span>
                  </div>
                  <div className="flex justify-between border-t pt-3 mt-3">
                    <span className="text-gray-900 font-medium">Price:</span>
                    <span className="font-bold text-blue-600">₹{selectedMentor.hourlyRate}</span>
                  </div>
                </div>
              </div>
              
              <button
                onClick={handleConfirmBooking}
                className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition duration-300"
              >
                Confirm Booking
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderFindMentorTab = () => {
    return (
      <div className="space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <svg className="w-4 h-4 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <input 
              type="text" 
              className="bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 p-2.5" 
              placeholder="Search by name, expertise, or company..."
              value={searchQuery}
              onChange={handleSearch}
            />
          </div>
          
          <div className="flex-shrink-0">
            <select 
              className="bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              {domains.map(domain => (
                <option key={domain} value={domain}>{domain}</option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {loading ? (
            Array(6).fill(0).map((_, idx) => (
              <div key={idx} className="bg-white p-4 rounded-lg shadow animate-pulse">
                <div className="flex items-center space-x-4">
                  <div className="rounded-full bg-gray-300 h-16 w-16"></div>
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                    <div className="h-3 bg-gray-300 rounded w-5/6"></div>
                    <div className="h-3 bg-gray-300 rounded w-1/2"></div>
                  </div>
                </div>
                <div className="mt-4 space-y-2">
                  <div className="h-3 bg-gray-300 rounded"></div>
                  <div className="h-3 bg-gray-300 rounded w-5/6"></div>
                  <div className="h-8 bg-gray-300 rounded mt-4"></div>
                </div>
              </div>
            ))
          ) : filteredMentors.length > 0 ? (
            filteredMentors.map(mentor => (
              <div key={mentor.id} className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300">
                <div className="p-5">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-xl overflow-hidden">
                        {mentor.profilePicture ? (
                          <img src={mentor.profilePicture} alt={mentor.name} className="w-full h-full object-cover" />
                        ) : (
                          mentor.name?.charAt(0) || "M"
                        )}
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-gray-900">{mentor.name}</h3>
                        <div className="flex items-center text-yellow-500">
                          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                          <span className="text-gray-900 ml-1">{mentor.rating}</span>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{mentor.title} at {mentor.company}</p>
                      <div className="flex items-center text-sm text-gray-500 mt-2">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {mentor.sessionsCompleted} sessions completed
                      </div>
                      <div className="flex items-center text-sm text-gray-500 mt-1">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        ₹{mentor.hourlyRate}/hour
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <p className="text-sm text-gray-600 mb-3">{mentor.bio}</p>
                    <h4 className="text-sm font-medium text-gray-900">Expertise</h4>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {mentor.expertise?.slice(0, 3).map((skill, idx) => (
                        <span key={idx} className="px-2 py-1 bg-blue-50 text-blue-700 text-xs font-medium rounded">
                          {skill}
                        </span>
                      ))}
                      {mentor.expertise?.length > 3 && (
                        <span className="px-2 py-1 bg-gray-50 text-gray-500 text-xs font-medium rounded">
                          +{mentor.expertise.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => handleBookSession(mentor)}
                    className="w-full mt-4 py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition duration-300"
                  >
                    Book a Session
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-full text-center py-12">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="mt-4 text-gray-500">No mentors found matching your criteria. Try adjusting your search.</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMySessionsTab = () => {
    const upcomingSessions = bookedSessions.filter(session => 
      session.status === "upcoming" && isAfter(parseISO(session.date), new Date())
    );
    
    const pastSessions = bookedSessions.filter(session => 
      session.status === "completed" || isBefore(parseISO(session.date), new Date())
    );

    return (
      <div className="space-y-8">
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upcoming Sessions</h3>
          {upcomingSessions.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {upcomingSessions.map(session => {
                const mentor = mentors.find(m => m.id === session.mentorId);
                return (
                  <div key={session.id} className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
                    <div className="p-5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-lg overflow-hidden">
                            {mentor?.name?.charAt(0) || "M"}
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{mentor?.name || "Unknown Mentor"}</h4>
                            <p className="text-sm text-gray-500">{mentor?.title} at {mentor?.company}</p>
                          </div>
                        </div>
                        <div className="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                          Upcoming
                        </div>
                      </div>
                      
                      <div className="mt-4 flex items-center text-sm text-gray-500">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        {format(parseISO(session.date), "MMMM d, yyyy 'at' h:mm a")}
                      </div>
                      
                      <div className="mt-2 text-sm text-gray-500">{session.topic}</div>
                      
                      <div className="mt-4 flex space-x-3">
                        <button className="flex-1 py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition duration-300">
                          Join Session
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8 bg-gray-50 rounded-lg">
              <p className="text-gray-500">You don't have any upcoming sessions.</p>
              <button 
                onClick={() => setActiveTab("findMentor")}
                className="mt-3 text-blue-600 hover:text-blue-700 font-medium"
              >
                Find a mentor
              </button>
            </div>
          )}
        </div>
        
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Past Sessions</h3>
          {pastSessions.length > 0 ? (
            <div className="space-y-4">
              {pastSessions.map(session => {
                const mentor = mentors.find(m => m.id === session.mentorId);
                return (
                  <div key={session.id} className="bg-white rounded-lg shadow-md overflow-hidden">
                    <div className="p-5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-lg overflow-hidden">
                            {mentor?.name?.charAt(0) || "M"}
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{mentor?.name || "Unknown Mentor"}</h4>
                            <p className="text-sm text-gray-500">{mentor?.title} at {mentor?.company}</p>
                          </div>
                        </div>
                        <div className="bg-gray-100 text-gray-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                          Completed
                        </div>
                      </div>
                      
                      <div className="mt-4 flex items-center text-sm text-gray-500">
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        {format(parseISO(session.date), "MMMM d, yyyy 'at' h:mm a")}
                      </div>
                      
                      <div className="mt-2 text-sm text-gray-500">{session.topic}</div>
                      
                      {session.notes && (
                        <div className="mt-3 p-3 bg-gray-50 rounded-md text-sm">
                          <span className="font-medium">Notes:</span> {session.notes}
                        </div>
                      )}
                      
                      <div className="mt-4 flex space-x-3">
                        <button className="flex-1 py-2 px-4 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium rounded-lg transition duration-300">
                          View Recording
                        </button>
                        <button className="py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition duration-300">
                          Book Again
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8 bg-gray-50 rounded-lg">
              <p className="text-gray-500">You don't have any past sessions.</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Find Your Tech Mentor</h1>
      
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-8">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <button
              onClick={() => setActiveTab("findMentor")}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === "findMentor"
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}
            >
              Find a Mentor
            </button>
            <button
              onClick={() => setActiveTab("mySessions")}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === "mySessions"
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}
            >
              My Sessions
            </button>
          </nav>
        </div>
        
        <div className="p-6">
          {activeTab === "findMentor" ? renderFindMentorTab() : renderMySessionsTab()}
        </div>
      </div>
      
      {renderBookingModal()}
    </div>
  );
};

export default MentorshipPlatform;