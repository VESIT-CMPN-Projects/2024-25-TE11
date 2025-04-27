import React, { useState } from "react";
import Icons from "./Icons";

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative flex items-center">
        <input
          type="text"
          placeholder="Search for jobs, skills, or companies..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-4 pl-12 pr-20 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
        />
        <div className="absolute left-4 text-gray-400">
          <Icons.Search className="w-5 h-5" />
        </div>
        {query && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              onSearch("");
            }}
            className="absolute right-20 text-gray-400 hover:text-gray-600"
          >
            <Icons.X className="w-5 h-5" />
          </button>
        )}
        <button
          type="submit"
          className="absolute right-3 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
        >
          Search
        </button>
      </div>
      <div className="flex flex-wrap gap-2 mt-3">
        <FilterButton label="Remote" icon={<Icons.Globe className="w-4 h-4 mr-1" />} />
        <FilterButton label="Full-time" icon={<Icons.Clock className="w-4 h-4 mr-1" />} />
        <FilterButton label="Posted this week" icon={<Icons.Calendar className="w-4 h-4 mr-1" />} />
        <FilterButton label="Salary" icon={<Icons.DollarSign className="w-4 h-4 mr-1" />} />
      </div>
    </form>
  );
};

const FilterButton = ({ label, icon }) => {
  const [selected, setSelected] = useState(false);

  return (
    <button
      type="button"
      onClick={() => setSelected(!selected)}
      className={`flex items-center px-3 py-1 rounded-full text-sm border transition-colors ${
        selected
          ? "bg-blue-100 text-blue-800 border-blue-300"
          : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
      }`}
    >
      {icon}
      {label}
    </button>
  );
};

export default SearchBar;