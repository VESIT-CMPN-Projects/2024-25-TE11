import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { doc, getDoc, collection, query, where, getDocs, updateDoc, addDoc } from "firebase/firestore";
import { db, auth } from "../firebase-config";

const MeetingRoom = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [participant, setParticipant] = useState(null);
  const [isMentor, setIsMentor] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [videoEnabled, setVideoEnabled] = useState(true);
  const [peers, setPeers] = useState([]);
  const { meetingId } = useParams();
  const navigate = useNavigate();
  
  // WebRTC references
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const localStreamRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const rtcSignalingChannelRef = useRef(null);
  
  useEffect(() => {
    const loadMeetingData = async () => {
      try {
        setLoading(true);
        const currentUser = auth.currentUser;
        
        if (!currentUser) {
          setError("You must be logged in to join a meeting");
          return;
        }
        
        // Find the session with this meeting ID
        const sessionsRef = collection(db, "sessions");
        const q = query(sessionsRef, where("meetingId", "==", meetingId));
        const querySnapshot = await getDocs(q);
        
        if (querySnapshot.empty) {
          setError("Meeting not found");
          return;
        }
        
        const sessionData = {
          id: querySnapshot.docs[0].id,
          ...querySnapshot.docs[0].data()
        };
        
        setSession(sessionData);
        
        // Determine if current user is the mentor or the student
        if (sessionData.mentorId === currentUser.uid) {
          setIsMentor(true);
          // Get student details
          const userDoc = await getDoc(doc(db, "userProfiles", sessionData.userId));
          setParticipant(userDoc.data());
        } else if (sessionData.userId === currentUser.uid) {
          setIsMentor(false);
          // Get mentor details
          const mentorDoc = await getDoc(doc(db, "employers", sessionData.mentorId));
          setParticipant(mentorDoc.data());
        } else {
          setError("You are not authorized to join this meeting");
          return;
        }
        
        // Initialize WebRTC
        initializeWebRTC(meetingId, currentUser.uid);
        
      } catch (error) {
        console.error("Error loading meeting:", error);
        setError("Failed to load the meeting. Please try again.");
      } finally {
        setLoading(false);
      }
    };
    
    loadMeetingData();
    
    // Clean up function for when component unmounts
    return () => {
      endVideoCall();
    };
  }, [meetingId]);
  
  // Initialize WebRTC
  const initializeWebRTC = async (roomId, userId) => {
    try {
      // Get local media stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      
      // Save stream reference
      localStreamRef.current = stream;
      
      // Display local video
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }
      
      // Initialize Firebase Realtime Database or Firestore for signaling
      setupSignalingChannel(roomId, userId);
      
      // Create peer connection
      createPeerConnection(stream, roomId, userId);
      
    } catch (err) {
      console.error("Error accessing media devices:", err);
      setError("Could not access camera or microphone. Please check permissions.");
    }
  };
  
  const setupSignalingChannel = (roomId, userId) => {
    // Use Firebase Firestore collection for signaling
    const signalingRef = collection(db, "signaling");
    
    // Create a document for this room if it doesn't exist
    // This is a simplified implementation. In a real app, you'd:
    // 1. Subscribe to changes on the signaling collection for this room
    // 2. Filter events not meant for this user
    // 3. Process "offer", "answer", and "ice-candidate" messages
    
    // For this demo, we'll manually listen for signaling messages
    // In a real implementation, you'd use Firebase onSnapshot for real-time updates
    
    rtcSignalingChannelRef.current = {
      // Send signaling message
      send: async (messageType, payload) => {
        try {
          await addDoc(signalingRef, {
            roomId,
            senderId: userId,
            type: messageType,
            payload,
            timestamp: new Date().toISOString()
          });
        } catch (err) {
          console.error("Error sending signaling message:", err);
        }
      },
      
      // Clean up function
      close: () => {
        // Unsubscribe from signaling events
        // (In a real app, you'd unsubscribe from your onSnapshot listener)
      }
    };
  };
  
  const createPeerConnection = (stream, roomId, userId) => {
    // STUN servers for NAT traversal - using Google's free STUN servers
    const configuration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    };
    
    // Create RTCPeerConnection
    const peerConnection = new RTCPeerConnection(configuration);
    peerConnectionRef.current = peerConnection;
    
    // Add local tracks to peer connection
    stream.getTracks().forEach(track => {
      peerConnection.addTrack(track, stream);
    });
    
    // Handle incoming remote stream
    peerConnection.ontrack = (event) => {
      if (remoteVideoRef.current && event.streams && event.streams[0]) {
        remoteVideoRef.current.srcObject = event.streams[0];
      }
    };
    
    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        // Send ICE candidate to the other peer via signaling channel
        rtcSignalingChannelRef.current.send('ice-candidate', {
          candidate: event.candidate
        });
      }
    };
    
    // Connection state changes
    peerConnection.onconnectionstatechange = () => {
      switch(peerConnection.connectionState) {
        case 'connected':
          console.log('WebRTC peers connected!');
          break;
        case 'disconnected':
          console.log('WebRTC peers disconnected');
          break;
        case 'failed':
          console.log('WebRTC connection failed');
          setError('Connection failed. Please try rejoining the meeting.');
          break;
      }
    };
    
    // Determine if this user should create the offer or wait for one
    // In a real app, you'd implement actual signaling logic here
    // For simplicity, let's say the mentor always creates the offer
    if (isMentor) {
      createAndSendOffer(peerConnection);
    } else {
      // Student listens for offer and sends answer
      // (In a real app, you'd implement these listeners in the setupSignalingChannel)
    }
    
    return peerConnection;
  };
  
  const createAndSendOffer = async (peerConnection) => {
    try {
      // Create offer
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      
      // Send offer via signaling channel
      rtcSignalingChannelRef.current.send('offer', {
        sdp: peerConnection.localDescription
      });
      
    } catch (err) {
      console.error("Error creating offer:", err);
      setError("Failed to establish video connection.");
    }
  };
  
  // In a real implementation, you'd have functions to handle incoming offers/answers
  // Such as handleOffer, handleAnswer, handleIceCandidate
  
  const toggleAudio = () => {
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = !audioEnabled;
      });
      setAudioEnabled(!audioEnabled);
    }
  };
  
  const toggleVideo = () => {
    if (localStreamRef.current) {
      localStreamRef.current.getVideoTracks().forEach(track => {
        track.enabled = !videoEnabled;
      });
      setVideoEnabled(!videoEnabled);
    }
  };
  
  const endVideoCall = async () => {
    // Stop all tracks in the local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
      localStreamRef.current = null;
    }
    
    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    
    // Close signaling channel
    if (rtcSignalingChannelRef.current) {
      rtcSignalingChannelRef.current.close();
      rtcSignalingChannelRef.current = null;
    }
  };
  
  const handleEndMeeting = async () => {
    try {
      if (isMentor && session) {
        // Update session status
        const sessionRef = doc(db, "sessions", session.id);
        await updateDoc(sessionRef, {
          status: "completed",
          endedAt: new Date().toISOString()
        });
      }
      
      endVideoCall();
      navigate(isMentor ? "/employer-dashboard" : "/employer-dashboard");
      
    } catch (error) {
      console.error("Error ending meeting:", error);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mx-auto"></div>
          <p className="mt-3 text-gray-700">Loading meeting room...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-center p-8 bg-white rounded-lg shadow-md">
          <svg className="w-12 h-12 text-red-500 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h2 className="mt-4 text-xl font-medium text-gray-900">{error}</h2>
          <button
            onClick={() => navigate(-1)}
            className="mt-4 py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition duration-300"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Meeting header */}
      <div className="bg-white shadow px-4 py-3 flex justify-between items-center">
        <div>
          <h2 className="text-xl font-medium text-gray-900">
            {session?.topic || "Mentoring Session"}
          </h2>
          <p className="text-sm text-gray-500">
            {isMentor ? "Session with " : "Learning from "}
            {participant?.name || "Participant"}
          </p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={handleEndMeeting}
            className="py-2 px-4 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition duration-300"
          >
            End Meeting
          </button>
        </div>
      </div>
      
      {/* Meeting content */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Video container */}
        <div className="flex-1 bg-gray-900 flex flex-col p-4">
          <div className="flex-1 flex items-center justify-center">
            {/* Remote video */}
            <div className="bg-gray-800 rounded-lg w-full max-w-3xl aspect-video flex items-center justify-center text-white overflow-hidden">
              <video 
                ref={remoteVideoRef} 
                autoPlay 
                playsInline 
                className="w-full h-full object-cover"
              />
              {!remoteVideoRef.current?.srcObject && (
                <p className="absolute">Waiting for other participant to join...</p>
              )}
            </div>
          </div>
          <div className="h-32 mt-4 flex items-center justify-center">
            {/* Local video */}
            <div className="bg-gray-700 rounded-lg h-28 w-40 flex items-center justify-center text-white overflow-hidden">
              <video 
                ref={localVideoRef} 
                autoPlay 
                playsInline 
                muted 
                className="w-full h-full object-cover"
              />
              {!videoEnabled && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-700 bg-opacity-70">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
              )}
            </div>
            
            {/* Video controls */}
            <div className="flex space-x-4 ml-6">
              <button 
                onClick={toggleAudio}
                className={`w-12 h-12 rounded-full ${audioEnabled ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'} flex items-center justify-center text-white transition-colors`}
              >
                {audioEnabled ? (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                  </svg>
                )}
              </button>
              <button 
                onClick={toggleVideo}
                className={`w-12 h-12 rounded-full ${videoEnabled ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'} flex items-center justify-center text-white transition-colors`}
              >
                {videoEnabled ? (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                  </svg>
                )}
              </button>
              <button 
                onClick={handleEndMeeting}
                className="w-12 h-12 rounded-full bg-red-600 hover:bg-red-700 flex items-center justify-center text-white"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.636 18.364a9 9 0 1012.728 0M12 14l9-5-9-5-9 5 9 5z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
        
        {/* Session information sidebar */}
        <div className="w-full md:w-80 bg-white border-l border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h3 className="font-medium text-gray-900">Session Info</h3>
          </div>
          <div className="p-4 space-y-4 overflow-y-auto flex-1">
            <div>
              <h4 className="text-sm font-medium text-gray-500">Topic</h4>
              <p className="text-gray-900">{session?.topic || "No topic specified"}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500">Duration</h4>
              <p className="text-gray-900">60 minutes</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500">Rate</h4>
              <p className="text-gray-900">${session?.hourlyRate || 0}/hour</p>
            </div>
            {session?.notes && (
              <div>
                <h4 className="text-sm font-medium text-gray-500">Notes</h4>
                <p className="text-gray-900">{session.notes}</p>
              </div>
            )}
            
            {/* Connection status information */}
            <div className="mt-6 p-3 bg-gray-50 rounded-lg">
              <h4 className="text-sm font-medium text-gray-500">Connection Status</h4>
              <div className="flex items-center mt-2">
                <div className={`w-3 h-3 rounded-full ${peerConnectionRef.current?.connectionState === 'connected' ? 'bg-green-500' : 'bg-yellow-500'} mr-2`}></div>
                <p className="text-sm text-gray-700">
                  {peerConnectionRef.current?.connectionState === 'connected' 
                    ? 'Connected' 
                    : 'Establishing connection...'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MeetingRoom;