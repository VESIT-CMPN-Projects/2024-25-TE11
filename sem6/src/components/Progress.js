const ProgressPage = () => {
    const progressData = {
      totalApplications: 25,
      interviews: 8,
      offers: 3,
      rejected: 5,
      monthly: [
        { month: 'Jan', applications: 5 },
        { month: 'Feb', applications: 8 },
        { month: 'Mar', applications: 12 },
      ],
      status: [
        { status: 'Applied', count: 9 },
        { status: 'Under Review', count: 6 },
        { status: 'Interview', count: 4 },
        { status: 'Offer', count: 3 },
        { status: 'Rejected', count: 5 }
      ]
    };
  
    return (
      <div className="p-6 max-w-4xl mx-auto">
        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-gray-500">Total Applications</h3>
            <p className="text-2xl font-bold">{progressData.totalApplications}</p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-gray-500">Interviews</h3>
            <p className="text-2xl font-bold text-blue-600">{progressData.interviews}</p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-gray-500">Offers</h3>
            <p className="text-2xl font-bold text-green-600">{progressData.offers}</p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-gray-500">Rejected</h3>
            <p className="text-2xl font-bold text-red-600">{progressData.rejected}</p>
          </div>
        </div>
  
        {/* Status Chart */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Application Status</h2>
          <div className="space-y-4">
            {progressData.status.map((item, index) => (
              <div key={index}>
                <div className="flex justify-between mb-1">
                  <span>{item.status}</span>
                  <span>{item.count}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${(item.count / progressData.totalApplications) * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
  
        {/* Monthly Progress */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Monthly Applications</h2>
          <div className="h-64 flex items-end justify-between">
            {progressData.monthly.map((data, index) => (
              <div key={index} className="flex flex-col items-center">
                <div 
                  className="w-16 bg-blue-500 rounded-t"
                  style={{ height: `${data.applications * 10}px` }}
                ></div>
                <p className="mt-2">{data.month}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

export default ProgressPage;