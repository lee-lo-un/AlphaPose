export default function NavButton({ onClick, bgColor, label }) {
  return (
    <button
      onClick={onClick}
      className={`${bgColor} hover:opacity-90 text-white font-bold py-4 px-6 rounded-lg 
        shadow-lg transition-transform hover:scale-105`}
    >
      {label}
    </button>
  );
}