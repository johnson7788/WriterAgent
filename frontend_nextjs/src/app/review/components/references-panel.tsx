import { FileText, Calendar, ExternalLink } from "lucide-react";
import React from 'react';

import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { ScrollArea } from "~/components/ui/scroll-area";

interface Reference {
  idx_val: number;
  file_id: string;
  title: string;
  publish_time: string;
  snippet: string;
  url: string;
}

interface ReferencesPanelProps {
  references: Record<string, Reference>;
  isLoading?: boolean;
}

const ReferencesPanel: React.FC<ReferencesPanelProps> = ({ references, isLoading }) => {
  const referencesList = Object.values(references).sort((a, b) => a.idx_val - b.idx_val);

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center text-lg">
            <FileText className="h-5 w-5 mr-2 text-blue-600" />
            参考文献
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-16 bg-gray-200 rounded"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (referencesList.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center text-lg">
            <FileText className="h-5 w-5 mr-2 text-blue-600" />
            参考文献
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
            <p>暂无参考文献</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3 flex-shrink-0">
        <CardTitle className="flex items-center text-lg">
          <FileText className="h-5 w-5 mr-2 text-blue-600" />
          参考文献
          <Badge variant="secondary" className="ml-2">
            {referencesList.length} 篇
          </Badge>
        </CardTitle>
        <p className="text-sm text-gray-600 mt-1">
          以下文献为撰写本大纲的参考资料
        </p>
      </CardHeader>
      <CardContent className="flex-1 p-0">
        <ScrollArea className="h-full px-6 pb-6">
          <div className="space-y-4">
            {referencesList.map((ref) => (
              <div
                key={ref.file_id}
                id={`ref-${ref.idx_val}`}
                className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:shadow-sm transition-all duration-200 scroll-mt-20"
              >
                <div className="flex items-start justify-between mb-2">
                  <Badge variant="outline" className="text-xs">
                    [{ref.idx_val}]
                  </Badge>
                  {ref.publish_time && (
                    <div className="flex items-center text-xs text-gray-500">
                      <Calendar className="h-3 w-3 mr-1" />
                      {ref.publish_time}
                    </div>
                  )}
                </div>
                
                <h4 className="font-medium text-gray-900 mb-2 text-sm leading-5 line-clamp-2">
                  {ref.title}
                </h4>
                
                <p className="text-xs text-gray-600 leading-relaxed line-clamp-2">
                  {ref.snippet}
                </p>
                
                <div className="mt-3 flex items-center justify-between">
                  <span className="text-xs text-gray-400 font-mono">
                    ID: {ref.file_id.slice(0, 8)}...
                  </span>
                  <a href={ref.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 text-xs flex items-center transition-colors">
                    <ExternalLink className="h-3 w-3 mr-1" />
                    查看详情
                  </a>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default ReferencesPanel;