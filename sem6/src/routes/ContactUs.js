import React, { useState } from 'react';

const ContactUs = () => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [subject, setSubject] = useState('');
    const [message, setMessage] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [submitted, setSubmitted] = useState(false);
    const [errors, setErrors] = useState({});

    const validate = () => {
        const newErrors = {};
        if (!name.trim()) newErrors.name = "Name is required";
        if (!email.trim()) newErrors.email = "Email is required";
        else if (!/\S+@\S+\.\S+/.test(email)) newErrors.email = "Email is invalid";
        if (!message.trim()) newErrors.message = "Message is required";
        
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!validate()) return;
        
        setSubmitting(true);

        const formSpreeEndpoint = 'https://formspree.io/f/xkgwaawv'; 

        const formData = new FormData();
        formData.append('name', name);
        formData.append('_replyto', email);
        formData.append('phone', phone);
        formData.append('subject', subject);
        formData.append('message', message);

        try {
            const response = await fetch(formSpreeEndpoint, {
                method: 'POST',
                body: formData,
                headers: {
                    Accept: 'application/json',
                },
            });

            if (response.ok) {
                setSubmitted(true);
            } else {
                console.error('Failed to submit the form. Please try again.');
            }
        } catch (error) {
            console.error('An error occurred while submitting the form.', error);
        } finally {
            setSubmitting(false);
        }
    };

    const handleBack = () => {
        window.history.back(); 
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-blue-100 p-4 sm:p-8">
            <div className="relative bg-white p-6 sm:p-8 rounded-2xl shadow-xl w-full max-w-lg border border-blue-100">
                {/* Decorative elements */}
                <div className="absolute -top-6 -left-6 w-12 h-12 rounded-full bg-blue-500 opacity-20"></div>
                <div className="absolute -bottom-6 -right-6 w-12 h-12 rounded-full bg-blue-500 opacity-20"></div>
                <div className="absolute top-0 right-0 w-32 h-32 rounded-bl-full bg-blue-50 -z-10"></div>
                
                <div className="mb-8 text-center">
                    <h2 className="text-3xl font-bold mb-2 text-blue-600">Contact Us</h2>
                    <p className="text-gray-600">We'd love to hear from you. Send us a message!</p>
                </div>
                
                {submitted ? (
                    <div className="py-8">
                        <div className="flex justify-center mb-4">
                            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                            </div>
                        </div>
                        <h3 className="text-2xl font-semibold text-center text-gray-800 mb-2">Thank You!</h3>
                        <p className="text-center text-gray-600 mb-6">Your message has been sent successfully. We'll be in touch with you shortly.</p>
                        <button 
                            onClick={handleBack} 
                            className="block w-full py-3 px-4 rounded-xl text-white bg-blue-600 hover:bg-blue-700 transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
                        >
                            Go Back
                        </button>
                    </div>
                ) : (
                    <form onSubmit={handleSubmit} className="space-y-5">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                                <input
                                    id="name"
                                    type="text"
                                    placeholder="Your name"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    className={`w-full p-3 rounded-xl border ${errors.name ? 'border-red-500' : 'border-gray-300'} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all`}
                                />
                                {errors.name && <p className="mt-1 text-sm text-red-500">{errors.name}</p>}
                            </div>
                            <div>
                                <label htmlFor="phone" className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                                <input
                                    id="phone"
                                    type="text"
                                    placeholder="Your phone number"
                                    value={phone}
                                    onChange={(e) => setPhone(e.target.value)}
                                    className="w-full p-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                                />
                            </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                                <input
                                    id="email"
                                    type="email"
                                    placeholder="Your email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    className={`w-full p-3 rounded-xl border ${errors.email ? 'border-red-500' : 'border-gray-300'} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all`}
                                />
                                {errors.email && <p className="mt-1 text-sm text-red-500">{errors.email}</p>}
                            </div>
                            <div>
                                <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
                                <input
                                    id="subject"
                                    type="text"
                                    placeholder="Subject of your message"
                                    value={subject}
                                    onChange={(e) => setSubject(e.target.value)}
                                    className="w-full p-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                                />
                            </div>
                        </div>
                        
                        <div>
                            <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">Message</label>
                            <textarea
                                id="message"
                                placeholder="Your message"
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                                className={`w-full p-4 rounded-xl border ${errors.message ? 'border-red-500' : 'border-gray-300'} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all`}
                                rows="5"
                            ></textarea>
                            {errors.message && <p className="mt-1 text-sm text-red-500">{errors.message}</p>}
                        </div>
                        
                        <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4 pt-2">
                            <button
                                type="submit"
                                className={`sm:w-2/3 py-3 px-6 rounded-xl text-white bg-blue-600 hover:bg-blue-700 transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5 ${submitting ? 'opacity-70 cursor-not-allowed' : ''}`}
                                disabled={submitting}
                            >
                                {submitting ? (
                                    <span className="flex items-center justify-center">
                                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Sending...
                                    </span>
                                ) : 'Send Message'}
                            </button>
                            <button
                                type="button"
                                onClick={handleBack}
                                className="sm:w-1/3 py-3 px-6 rounded-xl text-gray-700 bg-gray-100 hover:bg-gray-200 transition-all border border-gray-300"
                            >
                                Back
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
};

export default ContactUs;