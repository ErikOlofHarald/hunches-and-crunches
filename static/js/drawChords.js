/*
  The MIT License (MIT)

  Copyright (c) 2014 Steve Hall

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

var drawChords = function (data) {

  messages.attr("opacity", 1);
  messages.transition().duration(1000).attr("opacity", 0);

  matrix.data(data)
    .resetKeys()
    .addKeys(['left', 'right'])
    .update()

  var groups = container.selectAll("g.group")
    .data(matrix.groups(), function (d) { return d._id; });

  var gEnter = groups.enter()
    .append("g")
    .attr("class", "group");

  gEnter.append("path")
    .style("pointer-events", "none")
    .style("fill", function (d) { return colors(d._id); })
    .attr("d", arc);

  gEnter.append("text")
    .attr("dy", ".35em")
    .on("click", groupClick)
    .on("mouseover", dimChords)
    .on("mouseout", resetChords)
    .text(function (d) {
      var lastName = d._id.substr(d._id.indexOf(" "));
      return lastName;
    });

  groups.select("path")
    .transition().duration(2000)
    .attrTween("d", matrix.groupTween(arc));

  groups.select("text")
    .transition()
    .duration(2000)
    .attr("transform", function (d) {
      d.angle = (d.startAngle + d.endAngle) / 2;
      var r = "rotate(" + (d.angle * 180 / Math.PI - 90) + ")";
      var t = " translate(" + (innerRadius + 26) + ")";
      return r + t + (d.angle > Math.PI ? " rotate(180)" : " rotate(0)");
    })
    .attr("text-anchor", function (d) {
      return d.angle > Math.PI ? "end" : "begin";
    })
    .style("font-size", "12px")
    .style("cursor", "pointer");

  groups.exit().select("text").attr("fill", "orange");
  groups.exit().select("path").remove();

  groups.exit().transition().duration(1000)
    .style("opacity", 0).remove();

  var chords = container.selectAll("path.chord")
    .data(matrix.chords(), function (d) { return d._id; });

  chords.enter().append("path")
    .attr("class", "chord")
    .attr("d", path)
    .on("mouseover", chordMouseover)
    .on("mouseout", hideTooltip);

  chords.transition().duration(2000)
    .style("fill", function (d) {
      return colors(d.source._id);
    })
    .attrTween("d", matrix.chordTween(path));

  chords.exit().remove()


  function groupClick(d) {
    d3.event.preventDefault();
    d3.event.stopPropagation();
    addTag(d._id);
  }


  function chordMouseover(d) {
    d3.event.preventDefault();
    d3.event.stopPropagation();
    dimChords(d);
    var plural_source = (d.source.value.value == 1) ? " pass" : " passes";
    var plural_target = (d.target.value.value == 1) ? " pass" : " passes";
    tooltip
      .style("left", d3.event.pageX - 50 + "px")
      .style("top", d3.event.pageY - 70 + "px")
      .style("display", "inline-block")
      .html(
        d.source._id + " played " + d.source.value.value + plural_source + " to " + d.target._id +
        "<br>" +
        d.target._id + " played " + d.target.value.value + plural_target + " to " + d.source._id
      );
  }

  function hideTooltip() {
    d3.event.preventDefault();
    d3.event.stopPropagation();
    tooltip.style("display", "none")
    resetChords();
  }

  function resetChords() {
    d3.event.preventDefault();
    d3.event.stopPropagation();
    container.selectAll("path.chord").style("opacity", 1);
  }

  function dimChords(d) {
    d3.event.preventDefault();
    d3.event.stopPropagation();
    container.selectAll("path.chord").style("opacity", function (p) {
      if (d.source) { // COMPARE CHORD IDS
        return (p._id === d._id) ? 1 : 0.1;
      } else { // COMPARE GROUP IDS
        return (p.source._id === d._id || p.target._id === d._id) ? 1 : 0.1;
      }
    });
  }

};
