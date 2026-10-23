import type { PipeTransform } from '@angular/core';
import { Pipe } from '@angular/core';

import type { ImageElement } from '../../../interfaces/final-object.interface';

@Pipe({
  standalone: false,
  name: 'personFilterPipe'
})
export class PersonFilterPipe implements PipeTransform {

  constructor() { }

  /**
   * Return only the videos the selected person appears in. Membership is
   * read from the `persons` cache on each element (kept in sync with the
   * face index by FaceService); `refreshTrigger` re-runs the pipe when a
   * new index arrives and the cache may have changed.
   * @param finalArray
   * @param personId - person to filter by; null/'' disables the filter
   * @param refreshTrigger - bump to re-filter with unchanged personId
   */
  transform(finalArray: ImageElement[], personId: string, refreshTrigger: number): ImageElement[] {

    if (!personId) {
      return finalArray;
    }

    return finalArray.filter((element) => !!element.persons && element.persons.includes(personId));

  }

}
