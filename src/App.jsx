import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Home from "./pages/Home";
import Interview from "./pages/Interview";
import Report from "./pages/Report";
import Questions from "./pages/Questions";
import { getProfile } from "./lib/store";

function Guard({ children }) {
  return getProfile() ? children : <Navigate to="/" replace />;
}

export default function App() {
  return (
    <HashRouter>
      <div className="grain" />
      <Routes>
        <Route path="/" element={getProfile() ? <Navigate to="/home" replace /> : <Login />} />
        <Route path="/home" element={<Guard><Home /></Guard>} />
        <Route path="/interview" element={<Guard><Interview /></Guard>} />
        <Route path="/report" element={<Guard><Report /></Guard>} />
        <Route path="/questions" element={<Guard><Questions /></Guard>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  );
}
