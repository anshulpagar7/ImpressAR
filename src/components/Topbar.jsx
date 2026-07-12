import { Link, useNavigate } from "react-router-dom";

export default function Topbar({ minimal = false }) {
  const nav = useNavigate();
  return (
    <div className="topbar">
      <Link to="/home" className="logo">
        <span className="mark" />
        Impress<em>AR</em>
      </Link>
      {!minimal && (
        <div className="nav">
          <button className="btn btn-ghost" onClick={() => nav("/home")}>
            Home
          </button>
          <button className="btn btn-ghost" onClick={() => nav("/questions")}>
            Question bank
          </button>
          <button className="btn btn-brass" onClick={() => nav("/interview")}>
            Start interview
          </button>
        </div>
      )}
    </div>
  );
}
