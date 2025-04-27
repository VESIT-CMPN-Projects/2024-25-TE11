import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDQ1KzJbF-9938rllR2UQ_LuEdFsdmn7Gs",
  authDomain: "skillbridgeai-bd59d.firebaseapp.com",
  projectId: "skillbridgeai-bd59d",
  storageBucket: "skillbridgeai-bd59d.appspot.com",
  messagingSenderId: "137919663891",
  appId: "1:137919663891:web:2bcd156b6b81d2cc6b54e6",
  measurementId: "G-H2SYDTG83R"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
export { auth, app, db };