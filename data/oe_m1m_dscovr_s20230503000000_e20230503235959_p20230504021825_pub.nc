CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230503000000_e20230503235959_p20230504021825_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-04T02:18:25.090Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-03T00:00:00.000Z   time_coverage_end         2023-05-03T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx}�   �          @�{@��\@%?�=qAc�A�(�@��\?��H@
=A�(�A�                                      Bx}  �          @���@�(�@Q�?�=qA���Aٮ@�(�?�ff@-p�A�
=Az�\                                    Bx}�L  �          @��R@���@[�?.{@��BG�@���@0  @	��A���BQ�                                    Bx}��  
�          @��
@o\)@u<�>�p�B7  @o\)@Z=q?�G�A��B)z�                                    Bx}�Ę  �          @�(�@xQ�@h��?.{@��B,�\@xQ�@<(�@  A�  B��                                    Bx}��>  �          @�ff@�(�@J�H?�=qA�\)B  @�(�@�@5A��
A�
=                                    Bx}���  
�          @�z�@��H@E�?8Q�@��Bff@��H@�@�A��
A�p�                                    Bx}���  �          @�z�@��@S�
?�ffA4Q�B\)@��@�R@�A֏\A�
=                                    Bx}��0  
�          @��H@�Q�@Z�H?J=qA��B"{@�Q�@,��@  A�p�B                                      Bx}��            @���@W�@���<#�
>�BI33@W�@g�?�=qA���B<33                                    Bx}�|  �          @�
=@Mp�@�33=�?��BO�\@Mp�@g
=?���A�BA(�                                    Bx}�+"  
�          @�Q�@:=q@�zὣ�
�h��Ba@:=q@}p�?��A��RBV�                                    Bx}�9�  T          @�  @#�
@�녾�����Br�@#�
@�Q�?�
=A�G�Bk                                    Bx}�Hn  �          @�\)@!�@�녾���У�Bs�
@!�@�z�?�z�A�{BjG�                                    Bx}�W  �          @�@ ��@�Q쾞�R�[�Bs�@ ��@�p�?޸RA��
Bk�                                    Bx}�e�  �          @�p�@'
=@�ff�#�
��G�Bn=q@'
=@�  ?��HA�Q�Bc33                                    Bx}�t`  �          @�
=@��@��>u@'�By�@��@���@�Aϙ�Blz�                                    Bx}�  T          @��@(Q�@���=���?�=qBoQ�@(Q�@�  @
=A�ffBb                                    Bx}  �          @�p�@!G�@�Q�8Q��33Bs{@!G�@��?���A��
Bi��                                    Bx}�R  �          @��\@
�H@��\�.{��
=B�Q�@
�H@�p�?��A��Bz(�                                    Bx}��  "          @��H@��@��
    =L��B�(�@��@�(�@z�A��Bz33                                    Bx}～  v          @�G�@@��H    =L��B���@@��@33A�{B{��                                    Bx}��D  2          @��@	��@�  �#�
���B���@	��@�G�?�(�A��RBx{                                    Bx}���  "          @��R@p�@����
�h��B~  @p�@�  ?�z�A�
=Bt\)                                    Bx}��  "          @���@\)@������p�B}@\)@�=q?��A�Q�Bt�                                    Bx}��6  T          @��
@ff@��=L��?
=B��=@ff@���@�A�Q�B|                                    Bx}��  �          @��H@33@��<#�
>.{B��\@33@��@A��BQ�                                    Bx}��  T          @��H@
�H@�33>\)?��B�k�@
�H@��@
�HA�BwQ�                                    Bx}�$(  T          @���@Q�@�G�>L��@��B��R@Q�@~�R@(�AυBwQ�                                    Bx}�2�  �          @�\)@Q�@�  =�G�?�(�B�Q�@Q�@~�R@A��HBwp�                                    Bx}�At  �          @�\)@�\@�G�=���?�{B���@�\@���@ffA�B|�                                    Bx}�P  �          @���@33@��H=L��?�B��f@33@��H@�A�\)B}�                                    Bx}�^�  �          @�  @��@��=u?#�
Bp�@��@�  @�\A���Bt��                                    Bx}�mf  �          @��@��@��>.{?��B�@��@��@(�A�G�BvG�                                    Bx}�|  �          @���@
�H@���=u?#�
B��@
�H@�Q�@
�HA�33B{�                                    Bx}���  T          @�Q�?�Q�@�
=��=q�?\)B�\)?�Q�@��\?���A�ffB�ff                                    Bx}�X  �          @�
=?�p�@�����
�^�RB�{?�p�@�G�?��A���B�33                                    Bx}��  T          @�Q�?˅@��=L��?�B���?˅@��R@  A�\)B��                                    Bx}�  �          @���@@��H�#�
�#�
B�aH@@�33@�A�{B�                                    Bx}��J  �          @�Q�@�@��\<#�
>#�
B�k�@�@��\@��A�z�B��                                    Bx}���  
�          @��R?�G�@��ͼ��
��  B�ff?�G�@���@��A�  B�aH                                    Bx}��  T          @�ff?��@�=�Q�?s33B�G�?��@�z�@  A�33B��                                    Bx}��<  T          @���?���@�G�>8Q�?��RB�Ǯ?���@�\)@G�Aՙ�B�L�                                    Bx}���  �          @�z�?��H@�33=u?.{B��?��H@��\@(�A�p�B���                                    Bx}��  
�          @��?��@��H���
���B�z�?��@��@
=A���B�k�                                    Bx}�.  �          @��?��@�z����33B��)?��@��@�A�{B��                                    Bx}�+�  T          @��?�=q@���#�
��B�Q�?�=q@�@
=A�B��q                                    Bx}�:z  
�          @���?\@�p����
�.{B���?\@�@��A���B��                                    Bx}�I   �          @��\?���@�{�����k�B�Q�?���@��\?�\)A�
=B�ff                                    Bx}�W�  T          @���?��R@�
=��p���z�B���?��R@�(�?�A�=qB�                                    Bx}�fl  T          @�ff?޸R@�z�#�
��ffB��?޸R@�p�@ffA�G�B�{                                    Bx}�u  �          @��H?˅@��\����>�RB��\?˅@��R?�\)A��RB�                                    Bx}�  �          @��R?�  @��;L���{B���?�  @�  ?��HA���B�\)                                    Bx}�^  �          @�{?�@�������RB��?�@�\)@G�A��B�(�                                    Bx}�  �          @�?��@�33�#�
��(�B��?��@�{?�p�A��B�\                                    Bx}�  �          @�?ٙ�@�z὏\)�L��B��{?ٙ�@�@z�A�{B��                                    Bx}�P  �          @�?���@�p�        B��?���@�@��AîB�8R                                    Bx}���  �          @�ff?��
@�33�
=q���RB���?��
@�33?�\)A�B���                                    Bx}�ۜ  �          @�\)@   @�G�����33B��@   @�Q�?�z�A�
=B�8R                                    Bx}��B  T          @�{@�@�33�   ��  B|�@�@��?�ffA�\)Bwz�                                    Bx}���  �          @��R@p�@��H���Ϳ�{Bw\)@p�@�?�33A�G�Bn(�                                    Bx}��  �          @�{@&ff@�\)��  �1G�Bo�@&ff@�z�?�(�A��RBg��                                    Bx}�4  �          @�z�@(��@��<#�
>��Blz�@(��@~{?�A�Ba�\                                    Bx}�$�  T          @��
@�@��>u@*�HBv(�@�@|��@
=qA��
BiG�                                    Bx}�3�  T          @���@
=q@��=��
?h��B�=q@
=q@�p�@A��BzQ�                                    Bx}�B&  �          @�z�@�@�
=�u�8Q�B��@�@���?�p�A��B�\)                                    Bx}�P�  �          @��?�@�ff�����HB��{?�@��R?ǮA�z�B�z�                                    Bx}�_r  �          @��
@  @���������B~@  @��\?�  A��\Bz�                                    Bx}�n  �          @�@#33@�  �����S33Bq�R@#33@�?�A�ffBj�\                                    Bx}�|�  �          @�ff@
�H@�ff��Q쿆ffB�@
�H@���?�=qA��BwG�                                    Bx}�d  �          @��@��@�G�?uA*=qBx�
@��@l(�@4z�B�Bd                                      Bx}�
  �          @�@�@�{?.{@�Q�B�=q@�@|(�@(Q�A�(�Bx��                                    Bx}�  T          @��?�=q@�z�?��\Ad  B���?�=q@h��@HQ�B33B}{                                    Bx}�V  �          @��?�(�@�33?5Ap�B��R?�(�@vff@'�A��B{��                                    Bx}���  �          @��H@z�@���?p��A)�B��@z�@mp�@333B�Bs\)                                    Bx}�Ԣ  �          @�=q@Q�@�  ?E�A  B�k�@Q�@o\)@'�A��RBqz�                                    Bx}��H  "          @�\)@
=@��
>�\)@Mp�Bvff@
=@u�@�A��Bi\)                                    Bx}���  T          @�G�@��@�\)?
=@�\)Bp�@��@r�\@(�A��Bo                                    Bx}� �  �          @�
=@   @�\)��=q�L(�B���@   @�p�?�A��\B���                                    Bx}�:  �          @�@
=@��Ϳ   ���B��H@
=@�{?�
=A��B}\)                                    Bx}��  
�          @�{@ ��@�\)��{�~{Bm\)@ ��@~{?�  A�33Bf�                                    Bx}�,�  �          @�{@0  @��\��
=��z�B`(�@0  @w�?�{A33BZ�                                    Bx}�;,  �          @��R@0  @�33������B`��@0  @{�?��\AmG�B\(�                                    Bx}�I�  �          @��@��@�(�=�?�{B}Q�@��@z=q?�(�A�33Br�                                    Bx}�Xx  T          @���@   @�
=>�\)@QG�B��
@   @{�@	��A�=qB|(�                                    Bx}�g  �          @�p�@Q�@��>\@�\)B�k�@Q�@u�@{AָRBs�                                    Bx}�u�  �          @�p�@	��@���>�=q@K�B�@	��@w�@
=A�  Bs�H                                    Bx}�j  T          @��?�@�Q�>B�\@��B�?�@�Q�@A�G�B��f                                    Bx}�  �          @�z�@�@�(�=�?���B~G�@�@z�H?��HA��HBs��                                    Bx}�  "          @�(�@��@�33=u?@  B|@��@z=q?��A��Br�R                                    Bx}�\  T          @��H@G�@��
>�\)@W�B�aH@G�@vff@ffAΣ�By
=                                    Bx}�  �          @��H?�{@��R>8Q�@ffB�?�{@~{@�\A��
B���                                    Bx}�ͨ  
�          @��\@z�@�33>�{@���B��@z�@s33@��Aҏ\Bu�R                                    Bx}��N  �          @��
@Q�@��
=L��?z�B�{@Q�@|(�?��A�ffBv��                                    Bx}���  �          @��\@��@��\������B~��@��@~�R?ٙ�A��RBv�
                                    Bx}���  T          @��@33@��ͽ��Ϳ��B��@33@���?�G�A�{B|ff                                    Bx}�@  �          @��@p�@�녾aG��(��B{Q�@p�@�Q�?�{A���Btff                                    Bx}��  �          @���?�Q�@����Q�����B�?�Q�@��?�  A�=qB�p�                                    Bx}�%�  �          @���@�R@���.{��
Bx�@�R@z�H?�\)A��
Bqp�                                    Bx}�42  �          @�  @��@�p����
�tz�Bv  @��@z�H?���A���Bp                                      Bx}�B�  T          @��@�R@��W
=� ��Bw�R@�R@xQ�?ǮA�\)Bp��                                    Bx}�Q~  T          @�G�@!�@��<�>\BhG�@!�@k�?�(�A��HB]�H                                    Bx}�`$  T          @�G�@Fff@h��>�?\BF(�@Fff@P  ?У�A��B9z�                                    Bx}�n�  �          @���@U@X��>k�@1G�B5�@U@?\)?�\)A�
=B'33                                    Bx}�}p  �          @���@L��@b�\>�=q@Mp�B?�@L��@G
=?�(�A�ffB0��                                    Bx}�  �          @��H@333@|��>B�\@�BZ�@333@`��?���A�{BM                                    Bx}���  T          @��
@5�@}p�>k�@)��BY�H@5�@`��?���A�ffBL�\                                    Bx}��b  �          @�p�@.{@��H>���@`��Baff@.{@e?�(�A��RBS�                                    Bx}��  �          @�p�@333@���>u@4z�B\�H@333@dz�?��A���BO�\                                    Bx}�Ʈ  �          @�p�@@��@w�>�  @;�BP�\@@��@Z�H?�=qA�
=BB��                                    Bx}��T  
�          @�p�@a�@W�?\)@��HB.�\@a�@5?�z�A��\B��                                    Bx}���  �          @�{@K�@n�R>�Q�@��RBF=q@K�@P  ?�\)A��RB6z�                                    Bx}��  T          @��@333@���>.{@�\B\p�@333@e?���A��
BP{                                    Bx}�F  "          @�z�@333@�Q�=�\)?Q�B\\)@333@g�?��HA�=qBQQ�                                    Bx}��  "          @��\@#�
@��H=�\)?L��Bg@#�
@l��?޸RA�p�B]33                                    Bx}��  D          @��@�@�z�<�>���Bn\)@�@qG�?�p�A��\Bd��                                    Bx}�-8  2          @�z�@	��@���<��
>8Q�B@	��@�  ?���A�ffBw
=                                    Bx}�;�  
�          @���?��@��;Ǯ���B���?��@���?�ffA�ffB��
                                    Bx}�J�  T          @�=q?��
@�녾�{�vffB��R?��
@���?�z�A�
=B���                                    Bx}�Y*  S          @���?�@�\)�\)�˅B�#�?�@��
?�A���B�8R                                    Bx}�g�  �          @�  ?��H@�ff��G����\B��?��H@��\?���A�z�B��H                                    Bx}�vv  �          @�
=?���@��
�L�Ϳ!G�B��?���@��?�=qA�\)B�u�                                    Bx}��  "          @��R?�{@�33<��
>�  B��)?�{@�{?��A�\)B�                                    Bx}���  
�          @��?�Q�@�33>\)?˅B�\)?�Q�@���@   A��RB���                                    Bx}��h  T          @�33?�@�  =���?���B��{?�@���@G�A�
=B���                                    Bx}��  �          @��H?��H@�{����\)B��?��H@��H?��A��B�ff                                    Bx}���  �          @��?�
=@�
=����
=B�u�?�
=@��
?�ffA���B�B�                                    Bx}��Z  �          @�  ?�p�@���������\B�ff?�p�@�?\A�Q�B�k�                                    Bx}��   �          @�\)@�\@��׾����p�B��@�\@���?��HA�Q�B�Q�                                    Bx}��  
�          @��@p�@�p��u�+�Bs�R@p�@�z�?���A�ffBm33                                    Bx}��L  
}          @�=q@:=q@��ͼ��
�uB[@:=q@s33?�33A�ffBRG�                                    Bx}��  
�          @���@QG�@u=�?��BFz�@QG�@^{?�z�A��B:��                                    Bx}��  
�          @��H@P��@y��>B�\@�BH��@P��@`  ?޸RA�=qB<(�                                    Bx}�&>  
�          @�33@I��@�Q�=u?333BO{@I��@h��?�A��\BDp�                                    Bx}�4�  "          @�=q@H��@~{�8Q��33BN�@H��@mp�?�
=A��HBG
=                                    Bx}�C�  �          @�=q@A�@�녾�  �6ffBU�@A�@tz�?�33A�BNff                                    Bx}�R0  T          @�=q@:�H@�z᾽p���B[G�@:�H@|(�?���Ap��BU�                                    Bx}�`�  �          @���@<��@��������[�BW�@<��@u�?��AxQ�BQ��                                    Bx}�o|  �          @���@=p�@�=q��z��VffBW�@=p�@vff?���Ay�BQ�                                    Bx}�~"  
�          @��@<��@��H�u�+�BX�
@<��@vff?�A�{BR
=                                    Bx}���  c          @���@W�@mp��#�
��B?
=@W�@^{?�=qAvffB7G�                                    Bx}��n  �          @�\)@X��@hQ쾽p���G�B;��@X��@^�R?���ALQ�B6�                                    Bx}��  "          @�G�@`  @g
=��z��P  B7�R@`  @[�?�AW\)B1                                    Bx}���  T          @�
=@a�@^�R���
�j=qB2\)@a�@Tz�?��AJ�\B-                                      Bx}��`  �          @�G�@S33@s�
����>�RBDQ�@S33@fff?��\Ah��B=�                                    Bx}��  "          @�G�@N�R@w
=�u�.�RBH\)@N�R@h��?��Apz�BA�R                                    Bx}��  �          @��@\)@�p�    <#�
Bl�\@\)@u�?�33A��\Bc�
                                    Bx}��R  �          @�@!G�@�\)����p�Bl�R@!G�@|(�?�ffA���Be\)                                    Bx}��  T          @�z�@!�@�p��u�3�
Bk
=@!�@{�?�A�p�Bd�
                                    Bx}��  
�          @�z�@$z�@���u�1�Bi
=@$z�@z=q?�A���Bb�
                                    Bx}�D  
�          @�p�@-p�@�33��\)�N{Bbff@-p�@xQ�?�{A�{B\z�                                    Bx}�-�  �          @�33@/\)@�G�����=p�Be��@/\)@���?�Q�A�p�B_�                                    Bx}�<�  �          @���@333@�{����33B`�
@333@y��?��
A�
=BY=q                                    Bx}�K6  �          @�\)@<(�@{��B�\��RBT�H@<(�@l��?�{A��RBM��                                    Bx}�Y�  �          @�G�@i��@^{=��
?^�RB.�@i��@J=q?�
=A��B#��                                    Bx}�h�  �          @��@q�@P  =L��?\)B"��@q�@>{?��At  B��                                    Bx}�w(  �          @��\@q�@Y��    <�B'��@q�@HQ�?�=qAq�Bp�                                    Bx}���  "          @���@a�@fff�u�0��B6=q@a�@U?���Aw\)B-��                                    Bx}��t  �          @��H@e@fff>��@=p�B4=q@e@Mp�?��A�Q�B'Q�                                    Bx}��  T          @��H@hQ�@c�
�#�
����B1�
@hQ�@S33?���Av{B)                                      Bx}���  T          @�=q@g
=@c33��=q�C�
B1�@g
=@XQ�?�\)AL��B,=q                                    Bx}��f  �          @�=q@b�\@g��\)���
B6z�@b�\@X��?��\Ai�B.��                                    Bx}��  T          @��@]p�@p  ����{B=(�@]p�@`��?��Ar�HB5z�                                    Bx}�ݲ  �          @�z�@w�@XQ�>B�\@�B$@w�@C33?�p�A��B{                                    Bx}��X  �          @���@~{@QG�=�?���B  @~{@>{?�\)Av�HB\)                                    Bx}���  �          @���@p��@`�׽��
�p��B,(�@p��@Q�?��\Ad��B$33                                    Bx}�	�  �          @��@s33@^{>�z�@P  B)�@s33@Fff?���A��HB��                                    Bx}�J  �          @���@dz�@l��=�G�?�p�B7�@dz�@W�?\A�p�B-ff                                    Bx}�&�  "          @�{@U�@}p��u�&ffBH
=@U�@p  ?�ffAhz�BA�
                                    Bx}�5�  T          @�{@i��@j�H���Ϳ�\)B4\)@i��@[�?��Aj{B,��                                    Bx}�D<  �          @�p�@n{@e��L�Ϳ�B/��@n{@U�?���AmG�B'=q                                    Bx}�R�  �          @�{@b�\@q녾L����RB;��@b�\@dz�?�  A`(�B5(�                                    Bx}�a�  T          @��R@g
=@o\)�8Q���B833@g
=@a�?�  A_
=B1�                                    Bx}�p.  �          @���@w�@dz�=�Q�?xQ�B*��@w�@QG�?�Q�A}p�B �\                                    Bx}�~�  T          @���@���@X��=�Q�?�G�B G�@���@Fff?�\)Aq��BQ�                                    Bx}��z  
�          @��@�(�@U����Ϳ�\)B�@�(�@G�?�AK
=Bz�                                    Bx}��   
�          @��@s�
@l(��\)��(�B0G�@s�
@]p�?��\A]p�B)(�                                    Bx}���  �          @�G�@y��@c�
�.{����B)��@y��@W
=?�Q�AP  B#                                      Bx}��l  
�          @��@~�R@`�׾����B%�\@~�R@S�
?�
=AMG�B�H                                    Bx}��  "          @��@}p�@g
=��\)�B�\B)�\@}p�@W�?��A_�B!��                                    Bx}�ָ  �          @��H@w
=@k���\)�8Q�B.G�@w
=@[�?���Ae�B&p�                                    Bx}��^  T          @��
@y��@l(����
�Y��B-��@y��@\��?��Ab{B%��                                    Bx}��  T          @�(�@}p�@hQ������B*33@}p�@Z�H?��RAV{B#33                                    Bx}��  
Z          @�(�@{�@j�H�u�&ffB,=q@{�@[�?���Ab�RB$p�                                    Bx}�P  T          @�z�@|��@j�H�#�
��z�B+�@|��@]p�?�(�AR{B$�H                                    Bx}��  �          @�(�@�Q�@e��#�
�޸RB'33@�Q�@X��?�
=AK�B                                     Bx}�.�  �          @��H@�=q@\�;#�
��Q�B!33@�=q@P��?��AE�B�
                                    Bx}�=B  "          @��
@x��@k������P  B-��@x��@a�?��A:�HB)                                      Bx}�K�  T          @��\@}p�@c33������B'z�@}p�@U?���AQG�B ��                                    Bx}�Z�  T          @��H@{�@g
=�W
=�p�B*\)@{�@[�?�33AFffB$�                                    Bx}�i4  
�          @�33@w�@j�H��Q��z=qB.{@w�@c33?��\A/�
B*�                                    Bx}�w�  w          @��@h��@z�H��(����HB<ff@h��@s33?�ffA4  B8�                                    Bx}���  
�          @���@qG�@u��Q��tz�B5�H@qG�@l��?�=qA8��B1�R                                    Bx}��&  �          @��@{�@g
=������B*Q�@{�@`��?s33A#�B'                                      Bx}���  �          @�33@��H@Z=q�����z�B\)@��H@Tz�?^�RA�\Bp�                                    Bx}��r  �          @��@��R@R�\�Ǯ���RB�@��R@Mp�?W
=Az�B33                                    Bx}��  
�          @��H@�
=@9�����
��=qB��@�
=@-p�?��A3�A�p�                                    Bx}�Ͼ  �          @��\@���@?\)�B�\��\B��@���@6ff?k�A�\B�
                                    Bx}��d  �          @��@y��@`  ��=q�>{B'��@y��@W
=?��
A4  B"�H                                    Bx}��
  �          @��@j=q@u���z��H��B9{@j=q@j�H?�\)AC�B4G�                                    Bx}���  �          @���@g
=@w����
�Q�B;��@g
=@hQ�?��Aj{B4�                                    Bx}�
V  T          @�Q�@tz�@g
=�aG��ffB-�\@tz�@\(�?�{AB�HB(�                                    Bx}��  �          @�G�@j�H@s33���
�Z�HB8{@j�H@j=q?�=qA<��B3��                                    Bx}�'�  "          @�=q@k�@s�
���H���B7��@k�@n{?k�A (�B5Q�                                    Bx}�6H  T          @�33@i��@w
=������B:�@i��@r�\?aG�Az�B8=q                                    Bx}�D�  �          @��@Z=q@����Q��x��BIff@Z=q@}p�?��AE��BEQ�                                    Bx}�S�  �          @��@j=q@z�H�aG���B;��@j=q@n�R?��HAP(�B6p�                                    Bx}�b:  �          @���@g
=@���=�\)?5BC�
@g
=@vff?�ffA�G�B;=q                                    Bx}�p�  
�          @�
=@p  @y���Ǯ���B8z�@p  @q�?��
A/\)B4�                                    Bx}��  
�          @�=q@u@i����
=����B.ff@u@c�
?k�A
=B+z�                                    Bx}��,  "          @�33@~{@c33�������B'{@~{@`��?@  A��B%�H                                    Bx}���  �          @�\)@b�\@�p�����ffBF�@b�\@|(�?�\)AhQ�B@{                                    Bx}��x  
Z          @�=q@e@�  ����,(�BG
=@e@��?��\AR�RBA��                                    Bx}��  
(          @��\@mp�@��>.{?�p�BA  @mp�@u�?��A�z�B7z�                                    Bx}���  �          @�33@g�@�\)�L�;��BE@g�@~{?��HAs
=B>�\                                    Bx}��j  
�          @���@_\)@���>�(�@�p�BNff@_\)@}p�?�(�A�p�BB�\                                    Bx}��  
Z          @��R@^�R@�{?Q�A��BO�@^�R@w
=@A��B?��                                    Bx}���  T          @��\@XQ�@��?(�@���BP�R@XQ�@w
=@
=A�z�BC(�                                    Bx}�\  �          @�  @g
=@�z�<#�
>�BCz�@g
=@w�?�(�AxQ�B;��                                    Bx}�  �          @���@l(�@��
>�33@fffB@�R@l(�@o\)?��
A��B5z�                                    Bx}� �  �          @�33@xQ�@\)?\)@���B7�@xQ�@a�?�
=A�p�B)Q�                                    Bx}�/N  �          @��\@���@s�
?��@�(�B-ff@���@W
=?��A���B{                                    Bx}�=�  "          @���@w�@z�H>�33@hQ�B5��@w�@c33?ٙ�A�p�B*=q                                    Bx}�L�  T          @�33@j=q@}p�<��
>�  B=�@j=q@l��?�z�AqB5ff                                    Bx}�[@  �          @��@J�H@��ý�Q�c�
B\=q@J�H@���?�  A{�
BU��                                    Bx}�i�  �          @�=q@R�\@��>��@(Q�BV��@R�\@��
?�ffA���BMp�                                    Bx}�x�  �          @���@R�\@��>���@E�BT��@R�\@���?�A�\)BK
=                                    Bx}��2  �          @��\@p��@��>��R@L(�B>G�@p��@p  ?�(�A�\)B3�R                                    Bx}���  �          @���@fff@��R>�p�@tz�BE�@fff@tz�?�ffA���B:��                                    Bx}��~  �          @��H@)��@�z�>Ǯ@�  BvG�@)��@��R@33A���Bm{                                    Bx}��$  T          @�Q�@<(�@�z�>���@�ffBf�@<(�@��R?�p�A�ffB\ff                                    Bx}���  �          @�Q�@b�\@�{>Ǯ@���BG(�@b�\@s33?�A�p�B<
=                                    Bx}��p  �          @��R@fff@��H>��R@P  BB��@fff@o\)?ٙ�A��HB8ff                                    Bx}��  �          @���@[�@��>k�@�BN  @[�@~{?ٙ�A�p�BD�                                    Bx}���  E          @��H@n{@���>�33@dz�B@p�@n{@q�?�  A��
B5                                    Bx}��b  1          @��H@\(�@��
?��@�33BO\)@\(�@z�H@ ��A�Q�BC(�                                    Bx}�  �          @�33@_\)@��H?
=@��BL@_\)@xQ�@�A�z�B@
=                                    Bx}��  E          @�33@L(�@�Q�?W
=A
=BZ��@L(�@}p�@�A�
=BLz�                                    Bx}�(T  1          @�33@Y��@��?^�RA�
BP(�@Y��@s33@33A�ffB@��                                    Bx}�6�  
�          @�33@]p�@�=q?Tz�AQ�BM
=@]p�@q�@\)A�
=B>                                      Bx}�E�  "          @��H@G
=@�G�?n{A��B^G�@G
=@}p�@=qA�Q�BOQ�                                    Bx}�TF  
�          @��@`  @�?�@��RBHG�@`  @o\)?���A�ffB;�                                    Bx}�b�  �          @��@P  @��R?(�@���BW�@P  @�  @�A�p�BKQ�                                    Bx}�q�  
�          @�  @�@��H=��
?Y��B}=q@�@���?ٙ�A���Bw
=                                    Bx}��8  E          @�  ?�{@���#�
��33B�aH?�{@��?�=qA�(�B��\                                    Bx}���  c          @���@�\@��
��Q��r�\B��=@�\@�{?�\)Af{B�(�                                    Bx}���  �          @��@�@�{�\)��(�B�@�@�ff?\A���B}{                                    Bx}��*  "          @�
=@*�H@��ýu�.{Bs(�@*�H@���?��
A�(�Bm��                                    Bx}���  �          @��@\)@��R��33�n{Bxz�@\)@���?��RAW33BuG�                                    Bx}��v  �          @��
@�H@�  ������
B{�H@�H@��?��A6�HBz(�                                    Bx}��  �          @�{@5@�33��������Bi�@5@�
=?��AB�HBfG�                                    Bx}���  "          @��R@QG�@��
�8Q��z�BT�
@QG�@�?��\AX(�BP                                      Bx}��h  �          @��@XQ�@��\�����RBPG�@XQ�@�(�?���A^=qBJ�H                                    Bx}�  T          @�{@!�@�G���(���Q�Bxp�@!�@��?�
=AG�
Bu�
                                    Bx}��  "          @��R@�@�{��p��z=qB�z�@�@�G�?��
AXQ�B��                                    Bx}�!Z  �          @�
=@XQ�@��ýL�Ϳ   BN�H@XQ�@��?�{Ag33BH�H                                    Bx}�0   T          @�G�@r�\@��ý#�
����B;
=@r�\@tz�?��AV�HB4�H                                    Bx}�>�  "          @���@��\@n{=u?
=B)p�@��\@`  ?��\AT��B"ff                                    Bx}�ML  �          @�  @|��@u�=���?�ffB0z�@|��@e?���AbffB)                                      Bx}�[�  T          @�\)@z�H@u�=L��?��B1=q@z�H@fff?�ffAZ�HB*G�                                    Bx}�j�  
�          @�Q�@|��@vff=�G�?��B1{@|��@fff?���Ac�B)�\                                    Bx}�y>  "          @�G�@�=q@p      �#�
B*�@�=q@c33?�(�AK
=B$\)                                    Bx}���  
�          @�G�@�G�@s33���
�.{B-
=@�G�@fff?��HAJ�\B&�H                                    Bx}���  �          @��R@y��@tz��G���33B1z�@y��@i��?���A?
=B,G�                                    Bx}��0  �          @��@~{@r�\��\)�B�\B.��@~{@g
=?�33AA�B){                                    Bx}���  �          @��@�  @u���
�Q�B/Q�@�  @j�H?�z�AAG�B)�H                                    Bx}��|  �          @���@u@|�;�33�fffB7
=@u@vff?p��A��B4(�                                    Bx}��"  �          @�  @u�@{���z��?\)B7{@u�@tz�?}p�A%B3��                                    Bx}���  �          @�
=@w�@w
=�8Q��z�B3��@w�@mp�?��A3
=B/=q                                    Bx}��n  �          @�@o\)@{������ffB9��@o\)@qG�?�\)A=��B4�                                    Bx}��  �          @��@xQ�@xQ�#�
�\)B4�@xQ�@k�?�(�AN=qB.�                                    Bx}��  �          @���@��@k�>W
=@(�B'\)@��@[�?�\)AeB�                                    Bx}�`  �          @��@���@x�ý�Q�k�B0\)@���@n{?�33A=��B+(�                                    Bx}�)  �          @�(�@�z�@p��?\)@�Q�B(��@�z�@XQ�?�(�A��HB                                      Bx}�7�  �          @��\@�p�@i��?(��@أ�B$�@�p�@P��?��
A��\B
=                                    Bx}�FR  �          @��H@�{@hQ�?+�@�(�B#�R@�{@N�R?��
A��\B                                    Bx}�T�  T          @��\@�  @l��?�Q�AF{B+�@�  @HQ�@33A£�Bp�                                    Bx}�c�  �          @��
@�z�@l(�?k�A{B&�H@�z�@N{@�A�33Bff                                    Bx}�rD  T          @�{@���@g�?n{A33B \)@���@H��@G�A�=qB�R                                    Bx}���  "          @�ff@\)@���>��H@�
=B4�
@\)@j�H?�(�A��B*G�                                    Bx}���  
�          @�  @333@�G����
�uBs�H@333@���?�ffAyG�Bo                                      Bx}��6  �          @���@)��@�zὸQ�h��Bz�@)��@��?�G�Ar=qBv�                                    Bx}���  �          @�Q�@9��@�\)=�\)?0��Bo{@9��@��R?�{A��Bi�                                    Bx}���  �          @�
=@/\)@�  <��
>8Q�Bu(�@/\)@�  ?ǮA~�HBp{                                    Bx}��(  
Z          @�@1�@�p����R�J=qBr�@1�@���?�(�AG�Bo(�                                    Bx}���  "          @��\@-p�@��\�
=���
Br�H@-p�@���?k�A��Bq��                                    Bx}��t  "          @��
@0��@�녿333��
=Bp�@0��@���?O\)A��BpQ�                                    Bx}��  T          @��@@  @�\)��z��=p�Bf�\@@  @��H?�
=AABcp�                                    Bx}��  �          @�(�@;�@��þ����
=Bj{@;�@�?���A/�Bg�
                                    Bx}�f  "          @�z�@AG�@�  �\�z=qBf=q@AG�@�z�?��A1��Bc��                                    Bx}�"  �          @�
=@C33@�=q��Q��g
=Bf�@C33@�ff?���A5�Bc�                                    Bx}�0�  
�          @�\)@<��@��;L�Ϳ�p�Bk��@<��@�\)?�ffAR�\BhQ�                                    Bx}�?X  T          @��@=p�@��=u?(�Bk�\@=p�@��?�ffAz�\Bf33                                    Bx}�M�  �          @��
@@  @���=��
?L��Bm
=@@  @���?�{A~{Bg��                                    Bx}�\�  
�          @��@<(�@��    �#�
BoG�@<(�@�=q?��
Ar{Bjz�                                    Bx}�kJ  
�          @�=q@,(�@��;����Q�By��@,(�@��R?�z�AaG�Bu��                                    Bx}�y�  
�          @���@/\)@��
�u�ffBw�\@/\)@��R?��AQ�Btff                                    Bx}���  
�          @�33@Mp�@��\�����Z�HB[�@Mp�@�
=?��A-�BYG�                                    Bx}��<  �          @�{@>�R@��\��p��n�RBi(�@>�R@��R?��A1G�Bf                                    Bx}���  �          @�=q@4z�@�녾���Q�Bsz�@4z�@�
=?�=qA*�RBq��                                    Bx}���  "          @���@5@��þ��R�C33Bq��@5@�z�?���A@z�Bo=q                                    Bx}��.  
�          @�ff@,��@�Q쾏\)�333Bv��@,��@��
?�(�AF�\Bt                                      Bx}���  "          @�=q@2�\@�녾k��Q�Bo��@2�\@�p�?��HAHz�Bl�                                    Bx}��z  
�          @��@:=q@�Q�=�\)?+�Boff@:=q@���?�ffAx  Bj\)                                    Bx}��   
(          @�p�@4z�@�{>L��?�
=Bv  @4z�@���?�(�A��Bp\)                                    Bx}���  �          @�@0  @��>8Q�?޸RBy\)@0  @��R?�(�A��HBt                                      Bx~ l  "          @��R@1G�@���=�?�Q�By�@1G�@�  ?�A�(�Bt{                                    Bx~   �          @��@2�\@�{>��
@I��Bw  @2�\@��
?�=qA���Bp�R                                    Bx~ )�  �          @�z�@4z�@�z�?�@���Bu{@4z�@�  @33A���Bm=q                                    Bx~ 8^  "          @�@-p�@���>���@K�B{=q@-p�@�{?���A���Bu(�                                    Bx~ G  �          @��R@4z�@��>#�
?�G�Bv��@4z�@�
=?�
=A���Bq                                    Bx~ U�  T          @��@C�
@�z�>�=q@'�Bl��@C�
@��H?�  A�ffBf��                                    Bx~ dP  
�          @�  @AG�@�p�=u?\)Bn��@AG�@�{?ǮAp��Bj                                      Bx~ r�  �          @���@:�H@�Q�>.{?���Bs@:�H@�\)?�
=A��
Bnz�                                    Bx~ ��  T          @���@:=q@�Q�>�?��\Bt33@:=q@�  ?�33A~{Bo(�                                    Bx~ �B  T          @�G�@/\)@�(����Ϳn{B|
=@/\)@�?��HA_33Bx�                                    Bx~ ��  �          @��@O\)@��>���@:=qBa�@O\)@�=q?�
=A���BZ�\                                    Bx~ ��  T          @��\@q�@��
?L��@��BDz�@q�@}p�@G�A�B933                                    Bx~ �4  "          @�  @`��@���?�@���BP�@`��@�?�ffA�{BH(�                                    Bx~ ��  �          @��@P��@���>#�
?У�B_ff@P��@���?��
At(�BY                                    Bx~ ـ  T          @��@K�@���#�
�ǮBc�@K�@�ff?�p�AC�B_�                                    Bx~ �&  �          @�33@E�@������33Ba��@E�@�
=?��AU��B]�R                                    Bx~ ��  
�          @��H@a�@���>�G�@�Q�BJff@a�@�  ?��A���BB=q                                    Bx~r  �          @���@mp�@��?n{A33BA�@mp�@n�R@z�A���B4��                                    Bx~  T          @��R@c�
@�=q?�=qA/\)BI�@c�
@u@��A��\B<�                                    Bx~"�  
�          @�G�@^�R@��
?��HAj�\BM�
@^�R@q�@(Q�A��HB=�                                    Bx~1d  
�          @��@Tz�@�ff>��@,��BU�@Tz�@�ff?�  AzffBN�                                    Bx~@
  T          @�Q�@S�
@��?�@��BS�@S�
@���?�G�A��BJG�                                    Bx~N�  
Z          @�G�@Y��@�(�?��AiBJ=q@Y��@e�@\)A�ffB9�H                                    Bx~]V  	�          @���@W�@��?���A<  BN�@W�@p��@��A��B@�                                    Bx~k�  �          @���@S�
@���?�33A?�BP@S�
@q�@�\AÙ�BC(�                                    Bx~z�  T          @�33@H��@�=q?(�@�G�B]��@H��@�
=?���A�G�BU\)                                    Bx~�H  �          @�{@*=q@�33�Y����Bt�@*=q@�z�?�@�(�Bu�
                                    Bx~��  �          @���@0  @��R���\�#\)Bs��@0  @���>Ǯ@uBu�                                    Bx~��  "          @��\@4z�@�Q�p����
Br\)@4z�@�=q>�@�
=Bs��                                    Bx~�:  �          @���@4z�@�{�z�H�(�Bp�H@4z�@���>��@��
Brp�                                    Bx~��  T          @���@333@�ff����(z�Br  @333@���>�{@W
=Bs��                                    Bx~҆  T          @���@@  @�33�W
=��\Bh�@@  @�z�?�\@��Bi�H                                    Bx~�,  T          @�Q�@.�R@�Q�+���
=Bu�@.�R@�  ?8Q�@�Bu�\                                    Bx~��  
�          @���@\)@�Q�W
=��B��f@\)@�G�?!G�@ƸRB��                                    Bx~�x  �          @��@=p�@��ÿ�����Bn{@=p�@�\)?^�RA(�Bm�                                    Bx~  
�          @�33@0��@���.{���Bvz�@0��@�33?:�H@�p�Bv\)                                    Bx~�  "          @�G�@1�@�G���R���
Bt\)@1�@���?B�\@��HBs��                                    Bx~*j  T          @��H@<(�@��׾�G����\Bn�\@<(�@�ff?n{A=qBm33                                    Bx~9  T          @�33@HQ�@�p���33�\(�Bf  @HQ�@��\?z�HA�HBd(�                                    Bx~G�  
(          @��
@Vff@�=q�����B\�H@Vff@�?�33A4��BY                                    Bx~V\  �          @�p�@G�@��þ����K�Bh�
@G�@�?��
A�Bf�H                                    Bx~e  T          @�Q�@z=q@�{?
=q@�B;��@z=q@x��?��A�z�B3=q                                    Bx~s�  "          @�\)@l(�@��>�ff@�G�BF�H@l(�@��\?˅A�\)B?ff                                    Bx~�N  �          @��@aG�@�Q�>���@}p�BO�H@aG�@��?�=qA\)BI{                                    Bx~��  T          @�Q�@l��@�z�>�@���BG\)@l��@��?���A��B?��                                    Bx~��  
�          @�  @l��@��
?
=q@�BF�@l��@�=q?�
=A��B>�                                    Bx~�@  
�          @��R@h��@��?\)@�BHz�@h��@��?�Q�A��B@Q�                                    Bx~��  �          @�@j�H@���?�@���BF\)@j�H@�  ?�
=A�B>�                                    Bx~ˌ  "          @�ff@l(�@���?(�@�(�BEQ�@l(�@\)?��HA��B<�
                                    Bx~�2  T          @���@e�@��\?#�
@љ�BI�@e�@�Q�?�  A��\BA                                      Bx~��  �          @��
@aG�@��\?(��@���BK��@aG�@�Q�?�\A��RBB�                                    Bx~�~  T          @�z�@h��@�Q�?�@���BE�H@h��@~{?�z�A���B=�R                                    Bx~$  
�          @��
@hQ�@�Q�?\)@�
=BFp�@hQ�@~{?�33A�(�B>ff                                    Bx~�  T          @���@hQ�@���?�@�33BG�@hQ�@�Q�?�\)A�33B?ff                                    Bx~#p  T          @�(�@g�@���?�@���BF@g�@\)?���A�Q�B?�                                    Bx~2  
�          @��@j�H@�  ?�@�z�BD�R@j�H@~{?�{A�=qB<��                                    Bx~@�  "          @�\)@xQ�@�=q?��
A&�HB9ff@xQ�@j�H@�\A���B-z�                                    Bx~Ob  �          @�ff@z=q@\)?���A-�B6G�@z=q@e�@�
A��B)��                                    Bx~^  �          @�p�@tz�@\)?�p�AH(�B8��@tz�@c33@p�A�G�B+p�                                    Bx~l�  "          @�\)@k�@��\?E�@�\)BFz�@k�@\)?���A�ffB=33                                    Bx~{T  �          @�{@h��@���?n{A�RBF{@h��@x��?�p�A�\)B;�\                                    Bx~��  T          @�(�@_\)@��?s33A�
BL{@_\)@{�@ ��A��BA��                                    Bx~��  �          @��@�  ?���@"�\Ȁ\A��@�  ?�{@:�HA홚AH��                                    Bx~�F  �          @�z�@�=q?���@
=A��A�(�@�=q?�Q�@0  A���AS�
                                    Bx~��  
�          @�(�@��\?�
=@�A��A��\@��\?��\@1G�A��A5p�                                    Bx~Ē  T          @�  @��R@   @(�A�p�A�{@��R?��@'�AυAn�R                                    Bx~�8  
�          @���@�\)@
=@Q�A��
A��@�\)?�G�@%A��A���                                    Bx~��  T          @���@��R@��@�A��A�\)@��R?�
=@!�A�{A�
=                                    Bx~��  
�          @���@�{@��?��A�33A�{@�{?�33@�A��A��                                    Bx~�*  "          @��R@��@�R?�G�A��A�\)@��?�(�@�A�  A��\                                    Bx~�  �          @��@�33@��@�A���A��\@�33?У�@ ��A�
=A���                                    Bx~v  "          @���@�{@+�?�=qA�Q�A�@�{@��@�A\A�p�                                    Bx~+  "          @�(�@�=q@333?�A�A�33@�=q@��@{A���A��                                    Bx~9�            @���@�  @;�?�{A�\)A��@�  @��@!G�A�
=Aՙ�                                    Bx~Hh  �          @�Q�@�p�@C33?�Ac�
B�@�p�@'
=@�A��HA�                                    Bx~W  �          @�
=@���@G
=?\Av{B	Q�@���@(��@�RA���A�33                                    Bx~e�  E          @��@�33@L(�?���A�(�B��@�33@-p�@33A���A�G�                                    Bx~tZ  c          @��R@�@HQ�?ǮA���Bp�@�@*=q@G�A�
=B�                                    Bx~�   �          @���@�
=@'
=?��A>�HA��
@�
=@��?޸RA��Ạ�                                    Bx~��  T          @�G�@�ff@!G�?��RAy�A�{@�ff@@�
A�z�A��
                                    Bx~�L  
�          @�Q�@��H@^�R?��\AU�B!�@��H@Dz�@z�A�z�B�                                    Bx~��  �          @���@��@a�?�ABffB"�\@��@I��?�p�A�p�B�                                    Bx~��  �          @���@��@a�?�ffA/�
B"�R@��@K�?�\)A�=qB��                                    Bx~�>  "          @�p�@�G�@N�R?+�@�B{@�G�@>�R?�At  B33                                    Bx~��  �          @�{@�p�@\(�?\)@��B
=@�p�@Mp�?�{Ahz�B=q                                    Bx~�  T          @�  @�{@Z�H?fffAffB�@�{@Fff?�
=A���B{                                    Bx~�0  �          @��@�\)@a�?�@���Bz�@�\)@S33?�{Ab�HB                                      Bx~�  �          @��
@�
=@l��=L��>��B$�H@�
=@e�?k�A
=B!G�                                    Bx~|  �          @��H@o\)@\)?�=qA333B;�@o\)@g�?��RA��B0p�                                    Bx~$"  
(          @��H@w
=@|(�?\(�Ap�B6G�@w
=@hQ�?�G�A�
=B,��                                    Bx~2�  
�          @�Q�@p��@�Q�k��
=B;Q�@p��@|��?8Q�@���B9��                                    Bx~An  T          @�\)@o\)@�  =�Q�?z�HB;�R@o\)@w�?��
A,��B7�
                                    Bx~P  T          @��@{�@y��?=p�@��
B3
=@{�@g�?У�A�B*�                                    Bx~^�  �          @���@��\@s33?k�A{B+�
@��\@^�R?��
A��HB!�H                                    Bx~m`  
�          @���@xQ�@s�
?aG�A33B1�
@xQ�@`  ?޸RA�\)B(33                                    Bx~|  "          @�(�@qG�@mp�?}p�A)p�B2(�@qG�@XQ�?���A�=qB'z�                                    Bx~��  �          @�  @�=q@fff?c�
A��B%��@�=q@R�\?ٙ�A��B�H                                    Bx~�R  "          @�G�@��@c�
?J=qA�HB"\)@��@R�\?˅A�\)B=q                                    Bx~��  "          @�33@s33@n{?�R@�p�B1\)@s33@^{?��HA|��B)�                                    Bx~��  T          @�(�@hQ�@{�?�@�G�B=33@hQ�@mp�?�z�Ar�\B6�                                    Bx~�D  �          @�z�@}p�@{�?:�H@���B3(�@}p�@j=q?�{A�z�B+
=                                    Bx~��  "          @��H@u�@���?�@�z�B:  @u�@s�
?�
=AmG�B3ff                                    Bx~�  �          @��@h��@�=��
?G�BC@h��@��?��\A*=qB@G�                                    Bx~�6  �          @�@l(�@p��?c�
AB6{@l(�@]p�?�(�A�Q�B,�                                    Bx~��  �          @�\)@vff@c�
?���A�p�B+{@vff@G�@A�  B�H                                    Bx~�  �          @���@q�@n�R?��RA{33B2z�@q�@S33@33AĸRB$ff                                    Bx~(  �          @���@tz�@w�?�=qA4Q�B5\)@tz�@a�?�
=A��\B*                                    Bx~+�  
(          @���@r�\@���>�@��B;  @r�\@tz�?���AaG�B5                                      Bx~:t  �          @���@\)@p��>�(�@�
=B-ff@\)@dz�?�G�AS�
B'p�                                    Bx~I  T          @���@���@aG�?��\A*ffB!(�@���@L��?��
A�=qB��                                    Bx~W�  "          @��@y��@l(�?���A4Q�B-��@y��@W
=?�{A�33B"�                                    Bx~ff  �          @�\)@r�\@u�?uA!p�B5{@r�\@aG�?��
A�=qB+z�                                    Bx~u  T          @��
@s33@mp�?Q�A��B1{@s33@[�?�\)A��\B(Q�                                    Bx~��  �          @��
@qG�@p  ?8Q�@��RB3��@qG�@`  ?��
A��\B+�R                                    Bx~�X  �          @�z�@vff@mp�?+�@�(�B/�H@vff@^{?�(�A}�B(G�                                    Bx~��  
�          @�\)@�  @mp�?   @�  B+=q@�  @`��?�ffA\(�B$�H                                    Bx~��  �          @���@��H@l��>�p�@w�B(Q�@��H@a�?�AD  B"�                                    Bx~�J  �          @�=q@qG�@l��?#�
@�z�B2  @qG�@^{?�
=AyB*�                                    Bx~��  T          @���@l��@l��?&ff@��B3��@l��@^{?�Q�A}�B,��                                    Bx~ۖ  "          @�ff@g
=@l��?z�@�p�B6�
@g
=@_\)?�\)At��B0                                      Bx~�<  �          @�\)@n{@e�?Tz�A�RB/��@n{@S�
?˅A��B&��                                    Bx~��  T          @��@a�@l��?0��@��RB9�@a�@^{?�p�A��HB1�                                    Bx~�  �          @��@Z�H@q�?J=qA��B?33@Z�H@aG�?�=qA�
=B7{                                    Bx~.  �          @�\)@c�
@q�?�R@��HB:�@c�
@c�
?�A|(�B4                                      Bx~$�  �          @�G�@fff@q�?W
=A�HB9�R@fff@`��?У�A�  B1G�                                    Bx~3z  �          @��@e@tz�?W
=AffB:��@e@c33?��A�  B2��                                    Bx~B   �          @�G�@aG�@z=q?��@��B@�@aG�@l��?�\)Ap��B9�H                                    Bx~P�  
Z          @���@l��@k�?&ff@�z�B3z�@l��@]p�?�Az�HB,Q�                                    Bx~_l  
�          @��@tz�@k�?O\)A
ffB/��@tz�@Z�H?���A��B'p�                                    Bx~n  "          @�
=@z=q@o\)?@  @�{B.@z=q@_\)?��
A��
B'
=                                    Bx~|�  T          @�G�@|(�@tz�?8Q�@��B0�@|(�@e�?�G�A}�B)�                                    Bx~�^  T          @�
=@s33@w�?&ff@ٙ�B6{@s33@i��?���Au��B/G�                                    Bx~�  �          @�G�@w
=@z�H?(��@�33B5��@w
=@l(�?�(�Au��B.�H                                    Bx~��  �          @���@z�H@u?L��AG�B1�@z�H@e�?˅A��B)�R                                    Bx~�P  "          @�  @w
=@vff?�R@���B3�
@w
=@h��?�z�An�HB-=q                                    Bx~��  
Z          @���@e@qG�?G�A��B9�@e@a�?�ffA���B2                                      Bx~Ԝ  T          @��@qG�@l(�?n{A\)B1Q�@qG�@Z=q?�A�p�B(�\                                    Bx~�B  
�          @��@o\)@h��?n{A!G�B0��@o\)@W�?�z�A�  B'��                                    Bx~��  �          @�33@|��@\(�?���A=�B$=q@|��@H��?��A�  B�                                    Bx~ �  �          @�ff@z=q@h��?��\A,��B+@z=q@Vff?�  A��B"ff                                    Bx~4  
Z          @���@s33@l(�?���A7
=B0��@s33@X��?�ffA�=qB'                                      Bx~�  "          @��@n{@j�H?�
=AL(�B2�@n{@Vff?�z�A���B(�                                    Bx~,�  �          @��@p  @a�?�ffAc�B-
=@p  @L(�@   A��RB!�                                    Bx~;&  �          @�ff@���@G�?��A��
B(�@���@+�@�RA���B\)                                    Bx~I�  T          @��R@p��@B�\@�RA�p�B��@p��@{@C33BffB�                                    Bx~Xr  
�          @�\)@��\@Mp�?��HA��B�\@��\@2�\@�A�\)B	�                                    Bx~g  T          @�33@s33@hQ�?z�HA(z�B.�R@s33@W
=?�Q�A�\)B%�
                                    Bx~u�  �          @��
@\��@[�?У�A�  B3Q�@\��@A�@�Aә�B%=q                                    Bx~�d  T          @�z�@c�
@S33?�33A��\B+�@c�
@9��@�AӅB�H                                    Bx~�
  T          @���@Mp�@`  ?�\)A�Q�B>33@Mp�@G
=@�\A���B0�                                    Bx~��  
(          @�@k�@U?\A��B)  @k�@>{@	��A�  B�                                    Bx~�V  w          @��\@h��@a�?�\)A�p�B0ff@h��@H��@�\A˅B"��                                    Bx~��  
i          @��@mp�@c�
?�G�A��B/{@mp�@L(�@(�A���B"�                                    Bx~͢  T          @�
=@g
=@[�?�{A��HB.G�@g
=@C33@  A�  B �R                                    Bx~�H  w          @�@j�H@Z�H?��\Ac�B,
=@j�H@Fff?�A�z�B �                                    Bx~��  1          @��@c33@Z�H?���ApQ�B/��@c33@Fff?�(�A�B$z�                                    Bx~��  
�          @�=q@r�\@_\)?�ffAb�RB*�R@r�\@J�H?��HA�\)B��                                    Bx~	:  w          @��\@qG�@:�H?G�AG�BQ�@qG�@-p�?��A��HB�                                    Bx~	�  1          @�ff@aG�@>{?s33A:=qB �R@aG�@.�R?�G�A�  B(�                                    Bx~	%�  
�          @�{@~{@6ff?(�@�=qB�\@~{@+�?�33AX  B�
                                    Bx~	4,  
�          @��@�{@333>�G�@�
=B  @�{@*=q?xQ�A1��B�                                    Bx~	B�  �          @���@�z�@8��>Ǯ@��RBz�@�z�@0��?p��A$Q�B �R                                    Bx~	Qx  "          @�
=@P  @�?�z�AEBP�@P  @xQ�?���A�ffBH�                                    Bx~	`  �          @��@HQ�@��?�ffA1��BYQ�@HQ�@��\?��A�Q�BQ�H                                    Bx~	n�  �          @�ff@S33@���?E�Ap�BQ
=@S33@�G�?���A�=qBJ�                                    Bx~	}j  
�          @�  @[�@�  ?#�
@���BLff@[�@���?�Q�As
=BF��                                    Bx~	�  �          @�
=@[�@�\)?
=q@��
BK�H@[�@���?�=qAa�BF��                                    Bx~	��  �          @���@U@�G�?��@�Q�BI��@U@w
=?��Af�RBDz�                                    Bx~	�\  
�          @�33@XQ�@s�
?�@�=qBA�
@XQ�@h��?��RA_�B<�R                                    Bx~	�  
h          @��R@c�
@qG�?�R@ٙ�B:�\@c�
@e�?���Aj�RB4                                    Bx~	ƨ  �          @��@N{@z�H?�@��\BJ�\@N{@p  ?�  Ac33BE��                                    Bx~	�N  �          @��H@Tz�@tz�?
=q@ÅBD
=@Tz�@i��?�  Ad  B>�
                                    Bx~	��  
�          @�33@U�@u�?�@���BD
=@U�@j=q?�p�A^�HB?                                      Bx~	�            @��\@\(�@l��?�@�\)B<\)@\(�@aG�?�G�AeG�B6�H                                    Bx~
@  
6          @�(�@XQ�@s�
?   @��BA�@XQ�@i��?���AX��B<��                                    Bx~
�  F          @�z�@]p�@p��>��@�  B=�@]p�@g
=?�z�AQB8                                    Bx~
�  
6          @�p�@Z=q@u?
=q@�  BAff@Z=q@j�H?�  A_\)B<G�                                    Bx~
-2  
Z          @�
=@W�@|��?\)@�33BF(�@W�@q�?��
Ac
=BA{                                    Bx~
;�  
�          @��\@xQ�@g
=>�@�{B+@xQ�@]p�?��AFffB&�H                                    Bx~
J~  T          @�
=@��@N�R>���@G�B
=@��@G�?c�
AB=q                                    Bx~
Y$  
L          @��H@l(�@u>\@�(�B8z�@l(�@l��?��A;�
B4Q�                                    Bx~
g�  0          @�=q@z=q@c�
>�
=@�=qB)=q@z=q@Z�H?���A:=qB$�R                                    Bx~
vp  "          @��
@�Q�@P��>\@�33B��@�Q�@HQ�?xQ�A'33BQ�                                    Bx~
�  T          @��\@\)@G�>�Q�@��
BQ�@\)@@  ?n{A(��B                                      Bx~
��  �          @���@�33@<��?(��@��
B��@�33@1�?�
=AN�RB\)                                    Bx~
�b  
�          @��@��@Dz�?�R@�33B@��@:=q?�AO�B��                                    Bx~
�  
Z          @��@��\@S33?5@�
=B�R@��\@G
=?�ffAc
=BQ�                                    Bx~
��  
�          @�33@qG�@Z�H>u@/\)B(�@qG�@Tz�?\(�A\)B%�                                    Bx~
�T  
�          @�(�@w
=@Vff<#�
>.{B#@w
=@R�\?�R@޸RB!��                                    Bx~
��  �          @�@j=q@e�:�H��B1�@j=q@j=q��\)�B�\B3��                                    Bx~
�  �          @��R@�G�@N�R�\��B�R@�G�@P  >W
=@�
B=q                                    Bx~
�F  "          @��@��@L(�    �#�
B@��@H��?�@�B��                                    Bx~�  "          @�=q@�\)@@��?���A:�\B@�\)@1G�?˅A��
B�H                                    Bx~�  
�          @���@�
=@Dz�?fffA�\B	�@�
=@7
=?�
=Aq��Bz�                                    Bx~&8  T          @��@�(�@:=q?z�HA"�HB �@�(�@,(�?�p�Aw�
A�
=                                    Bx~4�  �          @�=q@�Q�@B�\?�Q�AF{B�
@�Q�@1�?��HA�p�A���                                    Bx~C�  "          @���@���@I��?�(�AK
=B33@���@8Q�?�G�A��B��                                    Bx~R*  �          @�=q@��@B�\?uA33B�R@��@4z�?�p�Aw\)A�                                      Bx~`�  �          @�z�@��@E?z�HA z�B(�@��@7�?�G�Ax��A���                                    Bx~ov  T          @���@�33@E?��A1��Bz�@�33@6ff?�{A���A�(�                                    Bx~~            @�@�{@Fff?O\)A
=B�H@�{@:=q?��AZffA��\                                    Bx~��  0          @���@�
=@AG�?+�@��BQ�@�
=@7
=?�Q�AB{A�33                                    Bx~�h  
�          @�@��@B�\?5@�{B��@��@7�?�p�AHQ�A�p�                                    Bx~�  T          @��@��@:=q?B�\@�Q�A�=q@��@.�R?�  AN=qA�                                    Bx~��  
�          @�33@�33@*�H?L��A�A���@�33@\)?�  AN�HAٙ�                                    Bx~�Z  	�          @�=q@�Q�@1G�?Q�A\)A�33@�Q�@%�?��AUA�                                    Bx~�   �          @��H@�{@:�H?Q�A\)A���@�{@.�R?��AYp�A�p�                                    Bx~�  "          @�33@�{@:=q?fffAz�A��\@�{@-p�?��Af=qA�=q                                    Bx~�L  "          @��@�(�@<��?^�RA(�B  @�(�@0��?�{Ac\)A�(�                                    Bx~�  T          @�
=@�z�?�\)?�G�A�z�A��@�z�?�=q?�ffA���A�G�                                    Bx~�  �          @�\)@��R?���?�A��As
=@��R?�G�@33A��A8z�                                    Bx~>  
�          @��\@��@6ff?fffA (�B�@��@)��?�\)Au�B {                                    Bx~-�            @�(�@G�@�녾�  �/\)BQ�H@G�@�G�>�ff@���BQff                                    Bx~<�  
�          @���@A�@��
�Ǯ���BVz�@A�@��
>��
@a�BV��                                    Bx~K0  T          @��@N{@��þB�\�Q�BM�@N{@�  >��H@��BL��                                    Bx~Y�  
�          @��@W
=@z=q>�?��RBEQ�@W
=@u�?J=qA��BC                                      Bx~h|  "          @��@e@s33>�\)@A�B:�\@e@l��?h��A ��B7z�                                    Bx~w"  
�          @��@��
@Z=q?
=@��B�@��
@P��?�z�AF�RB�\                                    Bx~��  �          @�  @�@,(�?uA!�A�@�@\)?�33Al(�A��                                    Bx~�n  �          @�{@�p�?�G�@
=A�z�A1�@�p�?#�
@  A���@�33                                    Bx~�  T          @�\)@�>��@ffA�
=@4z�@��\)@
=A��
C�8R                                    Bx~��  �          @���@��R���@�RA��
C�*=@��R�\)@�HAƏ\C���                                    Bx~�`  �          @���@�����
@��A���C���@����
=@�A�C��
                                    Bx~�  T          @���@��
��ff@'
=A�
=C���@��
�^�R@   A�G�C�+�                                    Bx~ݬ  
�          @�ff@�
=�(�@.�RA�\C�z�@�
=���@%A��C�                                      Bx~�R  	�          @���@�Q��@1�A癚C�g�@�Q�h��@*�HA�G�C��)                                    Bx~��  
�          @�Q�@��
�0��@>{A�
=C���@��
��
=@4z�A�RC�9�                                    Bx~	�  
�          @�Q�@�{�
=@9��A�C�� @�{��ff@0��A��C��)                                    Bx~D  
�          @�\)@�\)��33@1�A�z�C�  @�\)�L��@+�A�C�q�                                    Bx~&�  T          @��R@��<#�
@)��Aܣ�=��
@�녾�(�@'
=Aٙ�C��R                                    Bx~5�  
�          @��R@��\<�@'�A�{>���@��\����@%�A�G�C��q                                    Bx~D6            @�\)@��\�#�
@(��Aۙ�C�R@��\�
=@%�A�{C���                                    Bx~R�  
�          @��@�녾#�
@-p�A��C�
@�녿��@)��AۅC��                                     Bx~a�  �          @�
=@�G���G�@'
=A�z�C�}q@�G��\(�@ ��A��HC�'�                                    Bx~p(  "          @�Q�@��\��@��Aə�C��@��\��ff@�RA�=qC���                                    Bx~~�  T          @�ff@�=q�s33@��A�
=C��\@�=q��=q@��A���C��                                     Bx~�t  �          @�\)@�33�fff@(�A��
C��q@�33���
@��A�Q�C��                                    Bx~�  �          @�Q�@���R@�A�p�C���@���G�@�\A��C���                                    Bx~��  �          @��@�z���@\)A��C�aH@�z�^�R@Q�A�ffC�9�                                    Bx~�f  T          @�\)@������@$z�A�\)C�T{@���5@\)A�(�C��                                    Bx~�  T          @�{@�z��@��A�(�C�3@�z�fff@G�A�  C��                                    Bx~ֲ  "          @�ff@�녿G�@   A��C���@�녿�@A�  C�}q                                    Bx~�X  
Z          @��@��H�!G�@$z�A���C�w
@��H���
@(�A��C�C�                                    Bx~��  
�          @��R@����z�@%AׅC��)@����z�H@p�A�Q�C��                                     Bx~�  �          @��R@�G���@(Q�A�
=C�  @�G��p��@ ��A�z�C��R                                    Bx~J  �          @�ff@��׾�ff@*=qA�z�C�j=@��׿^�R@#33A���C�
                                    Bx~�  �          @��@�
=��{@,(�A���C��@�
=�B�\@&ffA���C��f                                    Bx~.�  �          @��@���?:�H@"�\A�\)A
{@���>��
@(Q�A�G�@r�\                                    Bx~=<  
Z          @��@���>��
@!�A���@vff@��׽��
@"�\A���C��                                    Bx~K�  T          @�z�@����(�@   AݮC�e@���Q�@��A�(�C�R                                    Bx~Z�  T          @�p�@�Q쾀  @#�
A�C�z�@�Q�&ff@\)AڸRC�)                                    Bx~i.  
�          @�{@�=�G�@0��A�ff?�{@����
@0  A���C��                                    Bx~w�  	�          @��@�{<#�
@,(�A�R=�Q�@�{���@*=qA뙚C�}q                                    Bx~�z  �          @���@���=��
@333A�33?p��@��׾�Q�@1�A��C�ٚ                                    Bx~�   �          @�p�@��H=�G�@<��A�33?�=q@��H��33@;�A�p�C��                                    Bx~��  �          @�z�@���>8Q�@>�RA��
@�@��þ�\)@>�RA�
=C�N                                    Bx~�l  
�          @���@��=L��@=p�A�33?�@�녾��@;�A�z�C��                                    Bx~�  T          @�{@�\)=#�
@J=qB�
?�@�\)��G�@G�B\)C�Q�                                    Bx~ϸ  T          @�p�@�\)���
@G�B�C��@�\)�   @E�B��C��)                                    Bx~�^  �          @��R@�(��\)@�
A�
=C���@�(��h��@��A��C�p�                                    Bx~�  
�          @��\@��R���@=qA���C�aH@��R�u@�\A�p�C�<)                                    Bx~��  �          @�Q�@��z�@A���C�xR@��n{@{AǮC�`                                     Bx~
P  �          @�@��H��@ffAי�C�@��H�^�R@\)A���C���                                    Bx~�  �          @�\)@��H���H@p�A�  C��
@��H�\(�@ffA�C���                                    Bx~'�  
�          @�  @��H�!G�@p�A�(�C��@��H��  @A��C�޸                                    Bx~6B  "          @��@�ff�L��@��A�33C�,�@�ff��z�@33A��HC�                                    Bx~D�  �          @�=q@��Ϳh��@��A��HC�~�@��Ϳ�G�@\)A�
=C�l�                                    Bx~S�  x          @�{@��׿B�\@<��B��C�5�@��׿���@333A��C��\                                    Bx~b4  
�          @�\)@�=q�B�\@:=qB �\C�G�@�=q��Q�@0��A�\C���                                    Bx~p�  T          @��@�z�fff@1�A�\)C���@�zῧ�@'
=A�33C�                                      Bx~�  
�          @��@�{�333@1�A���C�� @�{��{@(��A�C�T{                                    Bx~�&  T          @�G�@��׿�\@0  A�\C��
@��׿k�@(��A�Q�C���                                    Bx~��  �          @���@��ͿY��@8Q�A��RC��\@��Ϳ��\@.{A�G�C�T{                                    Bx~�r  �          @���@��
�0��@;�B 33C���@��
����@2�\A�
=C�(�                                    Bx~�  "          @���@��R���H@7
=A�
=C�
=@��R�h��@0  A��HC��                                    Bx~Ⱦ  �          @��@�Q�   @5�A���C��R@�Q�k�@.�RA���C���                                    Bx~�d  b          @�  @��þ�@+�A�p�C�Ff@��ÿY��@%�A�(�C��3                                    Bx~�
  
�          @�G�@��H��33@*=qA�(�C���@��H�=p�@%�A�z�C���                                    Bx~��  F          @�@����L��@�RA�Q�C���@�����
=@(�A���C��H                                    Bx~V  
6          @��H@�  �8Q�@0��B�C��f@�  �
=@,��B ��C��                                    Bx~�  �          @��@��\���@AG�B33C���@��\�k�@:=qBp�C���                                    Bx~ �  "          @�G�@���?:�H?�z�Aw\)A33@���?�\?�  A�  @���                                    Bx~/H  T          @��@�{=u@��A�{?J=q@�{��=q@  Ȁ\C�XR                                    Bx~=�  "          @�Q�@�(�=�\)@�RA�G�?k�@�(���z�@p�A߮C�0�                                    Bx~L�  �          @�G�@��R>aG�@
=AԸR@*=q@��R��@�A�\)C�1�                                    Bx~[:  T          @�  @�?��@ffA��@��@�>�\)@
=qA�{@^�R                                    Bx~i�  �          @�{@�(�?�\)?�{A�{A�\)@�(�?�=q@�\A�\)AQp�                                    Bx~x�  
�          @��@�{?�33?��Ahz�A���@�{?�Q�?�=qA�z�A�z�                                    Bx~�,  
�          @�Q�@�  ?�33?��HAUG�A���@�  ?��H?�p�A���A�{                                    Bx~��  �          @��@�z�@?O\)A  A�Q�@�z�?���?���AE�A���                                    Bx~�x  T          @��H@��@ff?L��A�A��
@��?��H?���AH  A�{                                    Bx~�  x          @�33@�(�?�?�\)AK33A�
=@�(�?У�?�\)Az=qA��                                    Bx~��  
�          @��@�?У�?�p�A_�A�p�@�?�
=?��HA���A�                                      Bx~�j  T          @��H@�
=?\?���ALz�A���@�
=?��?��As�A}                                    Bx~�  T          @�G�@�
=?��
?�G�A\z�Af�\@�
=?�=q?�
=A{\)AD(�                                    Bx~��  T          @��\@�\)?���?��
A���AB�\@�\)?W
=?�A�=qA�                                    Bx~�\  T          @�{@��R@{?�  A2=qA�@��R@�\?���Aq�A��
                                    Bx~  T          @�(�@��@ ��?:�HA  A�
=@��@�?��AD��A�z�                                    Bx~�  T          @���@�p�@"�\?@  A��A�\)@�p�@=q?�{AF�RA��                                    Bx~(N  
�          @�p�@�@"�\?J=qA��A���@�@��?�33AM�A�                                    Bx~6�  �          @�33@�G�@,(�?\)@���B G�@�G�@%�?p��A*{A�{                                    Bx~E�  "          @��\@�  @G�<��
>�=qB��@�  @E�>��@��Bz�                                    Bx~T@  �          @�p�@|��@U���Ϳ�=qB �@|��@Tz�>\@�Q�B��                                    Bx~b�  	�          @�(�@|��@QG��#�
��B�@|��@P��>��R@^{B(�                                    Bx~q�  
Z          @�Q�@u@Mp��L����
Bff@u@L��>��@?\)B=q                                    Bx~�2  �          @��@q�@N�R��ff��
=B!�@q�@P��<#�
>.{B#
=                                    Bx~��  
�          @��@{�@I�������33B�
@{�@K�=L��?z�B                                    Bx~�~  b          @�p�@~�R@Q녾�Q��~{B33@~�R@S33=�?��B��                                    Bx~�$  �          @��@{�@Tzᾮ{�uB �@{�@U�>�?��RB!33                                    Bx~��  
�          @��R@�Q�@S33��\)�Dz�B
=@�Q�@S�
>L��@
�HB=q                                    Bx~�p  
�          @�@���@N�R�W
=�
=B��@���@N{>�  @1�B�R                                    Bx~�  �          @��@���@>{<��
>.{Bp�@���@<(�>�G�@�{BG�                                    Bx~�  
�          @�33@��@4z�>8Q�@33B��@��@1G�?z�@�=qB�                                    Bx~�b  �          @��R@�ff@$z�?   @���A�p�@�ff@{?\(�A (�A�                                    Bx~            @��H@��
@"�\?��@ٙ�A�33@��
@�?uA-��A�ff                                    Bx~�  �          @�ff@�
=@&ff>��H@�z�A��H@�
=@   ?Y��A
=A�                                    Bx~!T  �          @�p�@�(�@,(�>��H@�
=A��R@�(�@&ff?^�RA33A�33                                    Bx~/�  �          @��@��@0��?
=@�ffB \)@��@)��?xQ�A+�A�=q                                    Bx~>�  �          @�@��
@,��?#�
@ᙚA�\)@��
@%�?�G�A4Q�A�ff                                    Bx~MF  �          @�z�@��\@*�H?333@��A���@��\@#33?���AA�A�                                    Bx~[�  �          @�Q�@���@!G�?(��@�A��
@���@��?�G�A9A�Q�                                    Bx~j�            @���@�Q�@(Q�?#�
@��HA�  @�Q�@!G�?�G�A8��A��R                                    Bx~y8  �          @���@�  @%�?@  A	�A�{@�  @��?�{ALQ�A�p�                                    Bx~��  �          @�  @�Q�@!G�?&ff@�  A���@�Q�@=q?�  A9�A�                                    Bx~��  �          @�z�@�ff@p�?\)@�33A��H@�ff@
=?h��A*ffA�Q�                                    Bx~�*  �          @���@\)@#33?&ff@���B�@\)@(�?�  AAp�A�\)                                    Bx~��  �          @�\)@�Q�@!G�?�@ÅA��H@�Q�@�H?aG�A"�HA���                                    Bx~�v  T          @��@�z�@'
=>���@^{B ��@�z�@"�\?(��@�  A��\                                    Bx~�  �          @�z�@�\)@�H>���@��A�ff@�\)@?=p�A�A�                                    Bx~��  �          @�@���@�?(�@��HA�R@���@��?p��A/\)A߮                                    Bx~�h  �          @���@��@p�?.{@��HA��@��@?��\A<��A�(�                                    Bx~�  �          @�=q@��
@p�?.{@��A�p�@��
@?��\A:=qA�                                    Bx~�  �          @���@�
=@��?8Q�AffA�\@�
=@G�?�ffAD��A�=q                                    Bx~Z  T          @�(�@�(�@\)?B�\A
=A�
=@�(�@
=?���APz�A�=q                                    Bx~)   T          @��@�Q�@�?333A��A�@�Q�@�
?��AJffA�\)                                    Bx~7�  �          @��@��H@�?=p�A\)A�
=@��H@	��?�ffAL��A�{                                    Bx~FL  �          @��R@���@z�?(��A ��A�ff@���@��?z�HA?�
A�ff                                    Bx~T�  �          @��@��@%?z�@�(�BG�@��@\)?p��A2�\A��
                                    Bx~c�  �          @�
=@\)@=q?+�A�RA�ff@\)@�\?�  AD(�A�ff                                    Bx~r>  �          @�p�@~{@8��>�
=@�{B
=@~{@333?Q�A{B�                                    Bx~��  �          @�\)@��@z�?Q�A�A�R@��@(�?���AS�A�G�                                    Bx~��  �          @��\@��\?\?�  A�ffA���@��\?��?�Q�A�Q�A��                                    Bx~�0  �          @��\@��
?�33?�  A�ffA�  @��
?�
=?�
=A���Ap��                                    Bx~��  �          @�G�@��?��?�p�A�(�A�Q�@��?��?�33A�\)Aa��                                    Bx~�|  �          @��@�Q�?�  ?�G�A��\A�\)@�Q�?��
?ٙ�A��A|��                                    Bx~�"  �          @�\)@�  ?�=q?��A��A��R@�  ?�\)?˅A�A�p�                                    Bx~��  �          @�\)@�=q@G�?s33A0��A�  @�=q@�?�G�Aj�HA��H                                    Bx~�n  �          @��@��R@\)?G�A�\A�\)@��R@
=?��AF�HA�ff                                    Bx~�  �          @�=q@�z�@�H?B�\A	��A���@�z�@�\?��AFffA�ff                                    Bx~�  �          @���@��\?��H?G�A�A���@��\?�=q?��A?33A���                                    Bx~`  �          @���@�
=@�?Tz�A(�A˅@�
=?��R?�\)AM�A��                                    Bx~"  �          @�  @��\?ٙ�?�ffABffA���@��\?��?��
AmG�A�=q                                    Bx~0�  �          @��@��\?��H?�  A9��A���@��\?�ff?�p�Ad��A��                                    Bx~?R  �          @��@�Q�@�R?5A=qA�@�Q�@
=?�ffAAA癚                                    Bx~M�  �          @���@�(�@�\?aG�A ��A�33@�(�@	��?�Q�AZ�HA��                                    Bx~\�  �          @���@�Q�@   ?^�RA�
A�
=@�Q�?�{?��ARffA��H                                    Bx~kD  �          @��@�G�@�?@  A	G�A�\)@�G�?��H?��A=��A���                                    Bx~y�  �          @�G�@��@�?8Q�A�A��@��@�
?��\A:�\A��H                                    Bx~��  �          @��@�z�@
=?Y��A�HA�=q@�z�@{?�AVffA֏\                                    Bx~�6  �          @�G�@��R@
=q?^�RA
=A�z�@��R@G�?�z�AUp�A�z�                                    Bx~��  �          @�@���@�?0��Ap�A��@���@p�?�G�A=��A�                                      Bx~��  �          @�p�@���@�?�R@��A�@���@
�H?n{A/�A֏\                                    Bx~�(  �          @���@���?�\)?p��A5A��@���?�(�?�Q�Ag\)A�Q�                                    Bx~��  �          @��
@�G�@ff?h��A,(�A�(�@�G�?���?�Q�Ac
=A�G�                                    Bx~�t  �          @�p�@��@{?O\)A��A�Q�@��@?�\)AR{A���                                    Bx~�  �          @�G�@�p�@-p�?Q�A��B  @�p�@$z�?�Q�AYA���                                    Bx~��  T          @���@�33@�H?J=qA�A��@�33@�\?�\)AM�A�=q                                    Bx~f  �          @�\)@��@ ��?(��@�A��H@��@��?�G�A;33A�\)                                    Bx~  �          @��@���@#33?333A Q�A��
@���@�?�ffA@��A�                                      Bx~)�  �          @��R@�=q@.{?@  A  B(�@�=q@%?���ARffB��                                    Bx~8X  �          @��@��@�R?z�HA7�
A��H@��@z�?��Ax��A�p�                                    Bx~F�  �          @�ff@���@/\)?@  AQ�B\)@���@'
=?���AS\)B�                                    Bx~U�  �          @��
@�=q@&ff?5A=qB�@�=q@{?���AJ=qA�{                                    Bx~dJ  �          @�Q�@��R?�  ?���A���A�z�@��R?��?�{A�ffA��\                                    Bx~r�  �          @���@l(�@9��?�  A@z�B�R@l(�@.�R?�33A�\)B{                                    Bx~��  �          @��@j�H@5�?xQ�A=�B�@j�H@*�H?�{A�33B��                                    Bx~�<  �          @��@n�R@&ff?��
AL��B��@n�R@�?��A���Bp�                                    Bx~��  
�          @�{@l��@0  ?Tz�A"�HB��@l��@'
=?��HAn�RB��                                    Bx~��  �          @���@o\)@(��?Tz�A#�
B�@o\)@\)?�Q�Al��B                                    Bx~�.  �          @�p�@q�@'�?L��AG�B
�@q�@�R?�z�Aep�B�                                    Bx~��  �          @���@l(�@#�
?8Q�A�B�@l(�@(�?���AY��B                                    Bx~�z  �          @�Q�@l��@�R?+�A��BQ�@l��@
=?�G�AO�B(�                                    Bx~�   �          @���@w�@ ��?#�
@��
B�R@w�@��?z�HAC33A��                                    Bx~��  �          @�33@w�@��?(�@���Bff@w�@?p��A<��A�p�                                    Bx~l  �          @��@l(�@33?(��A	�A�
=@l(�@�?z�HAM�A�z�                                    Bx~  �          @���@i��@ff?�@�
=B��@i��@\)?c�
A<z�A�Q�                                    Bx~"�  �          @���@[�@��?   @׮B33@[�@ff?W
=A6ffB	��                                    Bx~1^  �          @��H@P��@3�
>��H@���B"�
@P��@-p�?aG�A<Q�B�
                                    Bx~@  �          @�p�@W
=@333?�RAB\)@W
=@+�?�G�AT  B��                                    Bx~N�  �          @�{@b�\@%?+�A  B�@b�\@{?��
AW�Bz�                                    Bx~]P  �          @��@h��@*�H?+�A  B�@h��@#33?��AS�B�
                                    Bx~k�  �          @�(�@k�@.{?8Q�A=qB��@k�@%?��AZffB�                                    Bx~z�  �          @��
@j�H@.{?0��A	G�B��@j�H@%?���AU��B�
                                    Bx~�B  �          @��H@q�@{?G�AQ�B��@q�@?�\)AbffA���                                    Bx~��  �          @�\)@s�
@p�?:�HA�RA�G�@s�
@?��AV�HA�                                    Bx~��  �          @��@|(�@�
?z�HAF{A�z�@|(�?�33?�G�A�Q�Aͮ                                    Bx~�4  �          @�Q�@qG�@{?�ffAW�A��
@qG�@33?���A�  A�{                                    Bx~��  �          @�G�@}p�?�(�?}p�AIp�A�33@}p�?�?�G�A��HA�{                                    Bx~Ҁ  �          @��@���?�{?s33A?�A��@���?��H?�AmG�A�33                                    Bx~�&  �          @�
=@��H?˅?\(�A0z�A�(�@��H?���?���A^=qA���                                    Bx~��  �          @�ff@��
?�{?uAH��A���@��
?��H?�33Ap  A�G�                                    Bx~�r  �          @�@�
=?\(�?}p�AN{A733@�
=?5?���Af{A�
                                    Bx~  �          @���@�>aG�?�  AK
=@7�@�=��
?��\AO�?�                                      Bx~�  �          @��@�p��#�
?�\)A`��C��@�p��J=q?�G�AK�C��\                                    Bx~*d  �          @���@���aG�?���AQ�C���@�����
?uA6�\C�                                    Bx~9
  �          @�Q�@������?�Q�AQC��@����  ?s33A&�RC�
                                    Bx~G�  �          @���@�  ���?�
=AJffC���@�  ��ff?n{A\)C�)                                    Bx~VV  �          @�@�\)���?��RAT  C�  @�\)����?z�HA%p�C�Q�                                    Bx~d�  �          @��R@�\)��{?��\AVffC���@�\)�G�?z�HA&=qC��                                    Bx~s�  �          @�@�ff����?��\AX(�C��=@�ff��p�?}p�A(��C�
                                    Bx~�H  �          @��@�  ���H?��AB=qC��\@�  ��{?aG�A��C���                                    Bx~��  �          @��@�ff����?�
=AJ�RC���@�ff��(�?h��A33C�'�                                    Bx~��  �          @�z�@����Ǯ?��A9�C�c�@����ٙ�?Y��A��C���                                    Bx~�:  �          @�p�@�Q��G�?��A4��C�S3@�Q���?L��A
=C��                                    Bx~��  �          @��
@�(���(�?!G�@�{C��q@�(���ff>��@��\C���                                    Bx~ˆ  �          @�=q@�  ��Q�?(�@��
C��@�  ��\>�Q�@x��C�H�                                    Bx~�,  �          @��@�(��G�?k�A{C��@�(��Q�?�R@ӅC�5�                                    Bx~��  �          @�ff@�{�ff?fffA�C�xR@�{�p�?z�@�(�C���                                    Bx~�x  �          @�ff@��	��?aG�A  C�=q@��  ?��@��HC��)                                    Bx~  �          @��@���
�H?h��A(�C��
@����?z�@�\)C�N                                    Bx~�  �          @��\@��\�ff?L��A
ffC�@ @��\���>��H@�\)C�Ǯ                                    Bx~#j  �          @�{@�ff�z�?.{@���C�)@�ff�	��>�p�@��\C��R                                    Bx~2  �          @�
=@�z���?p��A&�\C�ff@�z��33?(�@׮C��3                                    Bx~@�  �          @�G�@���\)?���A;\)C�+�@����?:�H@�
=C���                                    Bx~O\  �          @�p�@�  �z�?�
=AH��C��)@�  �{?Q�A�C�Ff                                    Bx~^  �          @�  @��H��
?�Q�AH  C�<)@��H�p�?W
=A(�C��                                    Bx~l�  �          @�{@���  ?�z�AE�C�w
@�����?Q�A
{C�                                    Bx~{N  �          @��@�33��\?�AD  C�Z�@�33�(�?O\)Az�C���                                    Bx~��  T          @�\)@�����?��HAL  C�Ф@����!G�?W
=A�C�R                                    Bx~��  �          @�
=@���(�?�Q�AI�C��H@���?\(�AQ�C�%                                    Bx~�@  �          @�\)@��\�G�?�G�AT  C�l�@��\��?h��AQ�C��f                                    Bx~��  �          @�z�@�\)��?�p�AR�RC��@�\)�(�?aG�AC�\)                                    Bx~Č  �          @�Q�@������?��A\(�C���@����#�
?p��AG�C��=                                    Bx~�2  �          @�  @�����H?��AYG�C���@����%?k�Ap�C��H                                    Bx~��  �          @�p�@�ff��H?���AMp�C�b�@�ff�$z�?Tz�A�C���                                    Bx~�~  �          @��\@���ff?���Ap��C�XR@���!�?�G�A/�
C�w
                                    Bx~�$  �          @��@��H�
=?��RAW�C�c�@��H�!G�?^�RA�HC���                                    Bx~�  �          @��\@����
?�z�AHz�C��3@�����?L��A	C�)                                    Bx~p  �          @�(�@��� ��?��AC�
C��q@���)��?@  A z�C��                                    Bx~+  T          @�p�@��H�'�?�AG�C�#�@��H�1G�?B�\Ap�C�u�                                    Bx~9�  �          @��@����)��?��HAN�RC��H@����3�
?J=qA\)C�,�                                    Bx~Hb  �          @�(�@�G��)��?�Q�AK�C�� @�G��2�\?E�A(�C�/\                                    Bx~W  �          @�p�@���(��?��
AZ=qC���@���333?^�RA�HC�9�                                    Bx~e�  �          @�p�@����(��?���AeC��@����3�
?n{A{C�"�                                    Bx~tT  T          @�@��\�'�?�ffA]�C�R@��\�2�\?c�
A�HC�Q�                                    Bx~��  �          @�@�G��)��?���Al  C��H@�G��4z�?xQ�A#�
C��                                    Bx~��  T          @��@����333?�
=At��C�@����>�R?z�HA'�
C��                                    Bx~�F  �          @�@�ff�5?��RAR�RC���@�ff�@  ?J=qA��C�f                                    Bx~��  �          @��@����8��?�G�AW
=C�U�@����C33?L��AQ�C��H                                    Bx~��  �          @�(�@���O\)?}p�A+�C��q@���W
=>�@��RC�aH                                    Bx~�8  �          @��@a��?\)�
=q��{C��
@a��7��}p��EG�C�h�                                    Bx~��  �          @��H@^{�>{�u�C33C��H@^{�8�ÿ333�(�C��                                    Bx~�  �          @���@8Q��U���p����HC�s3@8Q��C�
��p��ɅC��                                    Bx~�*  �          @��R@C33�U��Q��k
=C�.@C33�G���Q���33C�+�                                    Bx~�  �          @�ff@���Z�H=�G�?���C��3@���X�þ�
=��
=C���                                    Bx~v  �          @��
@�  �U�>��@��C��)@�  �W��L�Ϳ�\C���                                    Bx~$  �          @��H@�
=�P��?J=qA�RC���@�
=�Vff>�\)@6ffC���                                    Bx~2�  �          @�Q�@�z��R�\?�@�
=C��
@�z��U�<#�
=��
C�j=                                    Bx~Ah  �          @�  @�p��QG�>�z�@@  C�@�p��QG��aG��C��q                                    Bx~P  �          @�G�@���=p�?W
=A
�HC���@���C�
>�Q�@p��C�5�                                    Bx~^�  �          @���@���7�?^�RA��C�.@���>{>��@���C��)                                    Bx~mZ  �          @���@���8Q�?\(�A�\C�+�@���>�R>���@��C���                                    Bx~|   �          @���@�ff�333?G�A{C�� @�ff�8Q�>���@\��C�:�                                    Bx~��  �          @���@�\)�2�\?!G�@ҏ\C��
@�\)�7
=>B�\?�
=C�l�                                    Bx~�L  �          @�G�@��H�@  ?G�A��C�k�@��H�E�>���@E�C��                                    Bx~��  T          @�G�@�Q��0  ?�@��
C���@�Q��3�
=�?��\C��)                                    Bx~��  �          @��@��\�AG�?p��A\)C�L�@��\�HQ�>�ff@�(�C��3                                    Bx~�>  �          @�=q@�z��:�H?fffAC��=@�z��A�>�(�@�C�s3                                    Bx~��  �          @�G�@�Q��C33?��\A*�\C���@�Q��K�?�@�{C�j=                                    Bx~�  �          @�=q@�ff�Mp�?z�HA!p�C�R@�ff�Tz�>�ff@���C��q                                    Bx~�0  �          @���@��
�QG�?aG�A�\C��)@��
�W�>�33@fffC�33                                    Bx~��  �          @��\@����H��?s33Ap�C���@����P  >�G�@���C�(�                                    Bx~|  �          @���@�p��E?�
=AEC��H@�p��O\)?+�@�
=C��q                                    Bx~"  �          @��@�{�HQ�?���AF�\C�ff@�{�Q�?.{@�\)C�                                    Bx~+�  �          @��@��J�H?��A5�C�7
@��S33?\)@��HC��                                    Bx~:n  �          @���@���L(�?���A2=qC�\@���Tz�?
=q@��C���                                    Bx~I  �          @���@�(��I��?�(�ALQ�C�(�@�(��S33?333@�\)C�~�                                    Bx~W�  �          @��@���K�?�=qA\��C��R@���Vff?J=qA
=C�=q                                    Bx~f`  �          @�33@�p��XQ�?���A�z�C��H@�p��e?��
A(��C���                                    Bx~u  �          @��H@�  �U?�33Ah��C��@�  �aG�?W
=A
=qC�.                                    Bx~��  �          @��\@��R�W�?�z�Ak
=C���@��R�c�
?W
=A33C��                                    Bx~�R  �          @��\@�Q��G
=?aG�AffC���@�Q��Mp�>�Q�@o\)C�J=                                    Bx~��  �          @��@��H�W
=?W
=A33C�  @��H�\��>�\)@6ffC��                                     Bx~��  �          @��\@����[�?z�HA"�RC���@����c33>��@�p�C�.                                    Bx~�D  �          @�=q@���Z�H?���A7
=C���@���c33?�@���C�f                                    Bx~��  �          @�33@�p��Tz�?W
=A
�HC���@�p��Z�H>�\)@9��C�%                                    Bx~ې  �          @�33@�{�e?uA�C��{@�{�l��>�33@fffC�Ff                                    Bx~�6  �          @��
@�G��`  ?fffA�C�` @�G��fff>���@G�C���                                    Bx~��  �          @��H@��H�l��?\(�A=qC��@��H�r�\>k�@��C��{                                    Bx~�  �          @��H@�(��h��?s33AQ�C�G�@�(��p  >���@VffC�޸                                    Bx~(  �          @�33@����_\)?Tz�A(�C�xR@����e�>k�@Q�C�q                                    Bx~$�  �          @��\@��H�XQ�?fffA��C��@��H�^�R>��
@S�
C��H                                    Bx~3t  �          @�G�@����XQ�?c�
Az�C��\@����^�R>��R@N�RC��                                    Bx~B  �          @�G�@�(��L(�?���A<  C���@�(��U�?�@�{C�Z�                                    Bx~P�  �          @��@�
=�`  ?^�RA��C�*=@�
=�fff>�=q@1�C��f                                    Bx~_f  �          @���@��R�]p�?h��A�
C�G�@��R�dz�>��R@N{C��)                                    Bx~n  T          @��@���^�R?\(�A
=C�S3@���e�>��@+�C��                                    Bx~|�  �          @��H@�Q��`  ?k�A�HC�G�@�Q��g
=>��R@H��C��)                                    Bx~�X  �          @��@�
=�_\)?s33A{C�8R@�
=�fff>�{@c33C��f                                    Bx~��  �          @��@�(��fff?n{A{C�|)@�(��l��>���@FffC��                                    Bx~��  �          @��@�G��hQ�?O\)A33C�@�G��n{>.{?��C���                                    Bx~�J  �          @���@����l��?O\)A{C���@����q�>#�
?У�C�l�                                    Bx~��  �          @���@�  �n{?O\)AffC���@�  �s�
>��?˅C�5�                                    Bx~Ԗ  �          @�  @|(��n�R?c�
AC�G�@|(��u�>u@��C��                                    Bx~�<  �          @�  @}p��n�R?Q�AQ�C�\)@}p��tz�>#�
?�\)C�
=                                    Bx~��  �          @�=q@}p��s�
?^�RA�
C�@}p��y��>B�\?��RC��{                                    Bx~ �  �          @��@y���w
=?^�RA  C��q@y���}p�>8Q�?��C�Ff                                    Bx~.  �          @���@����l��?L��AC��\@����q�>\)?��RC�^�                                    Bx~�  �          @�G�@��\�k�?=p�@��C��)@��\�p  =��
?Y��C��{                                    Bx~,z  �          @���@|(��u�?E�@�
=C��@|(��z=q=��
?\(�C��q                                    Bx~;   �          @�=q@~{�u�?=p�@�33C���@~{�y��=L��?�\C��)                                    Bx~I�  �          @�Q�@�=q�j=q?�@�{C�H@�=q�mp���Q�fffC���                                    Bx~Xl  �          @�Q�@����l(�?
=q@�p�C���@����n�R����p�C���                                    Bx~g  �          @���@x���xQ�?
=q@�(�C��=@x���z�H�#�
��\)C�j=                                    Bx~u�  �          @�G�@�p��h��>���@��
C�j=@�p��j=q����)��C�]q                                    Bx~�^  �          @��\@�ff�i��>���@X��C���@�ff�i�������W�C���                                    Bx~�  �          @���@���p��>\@z=qC���@���qG������G
=C��\                                    Bx~��  �          @�Q�@g
=���
>��H@�33C��q@g
=���;���,��C��=                                    Bx~�P  �          @���@a����>��R@N�RC��@a���
=�����C��                                    Bx~��  �          @�G�@i�����>B�\?�C��@i�����
����z�C���                                    Bx~͜  �          @�  @xQ��w�>�{@dz�C��@xQ��w���Q��s�
C��f                                    Bx~�B  �          @��@mp���G�>��R@QG�C�C�@mp����þ�(���{C�L�                                    Bx~��  �          @�@l���}p�=���?��C��@l���z=q��R�ӅC���                                    Bx~��  �          @�(�@s33�s33�W
=�p�C��@s33�l(��fff���C���                                    Bx~4  �          @��H@_\)��Q쾏\)�>�RC�� @_\)�x�ÿ�G��.ffC��                                    Bx~�  �          @�33@b�\��Q�.{���
C���@b�\�z=q�fff��\C��                                    Bx~%�  �          @�{@P����Q�\��p�C���@P���w���{�E��C�                                    Bx~4&  T          @��\@Z=q�o\)������
C�,�@Z=q�fff����F�RC��R                                    Bx~B�  �          @�
=@e�q녾�\)�C33C�� @e�j=q�xQ��+�
C�33                                    Bx~Qr  �          @�G�@\)�Z�H>W
=@33C��3@\)�Y����(�����C���                                    Bx~`  �          @�=q@���Q�?��@���C��q@���U���\)�G�C��                                    Bx~n�  �          @��\@�=q�W�?0��@�z�C�33@�=q�\(�=#�
>�C��=                                    Bx~}d  �          @���@y���aG�?�@�Q�C��
@y���c�
�\)���RC�Ф                                    Bx~�
  �          @�
=@n�R�h��>�\)@FffC�� @n�R�g������ffC��                                    Bx~��  �          @�z�@^�R�qG��W
=��
C�aH@^�R�j=q�h���#�C��=                                    Bx~�V  �          @�(�@W��u��\)�J�HC��q@W��n{��  �4��C�3                                    Bx~��  �          @��@\���z�H�L����C���@\���s�
�p���%�C�R                                    Bx~Ƣ  �          @��R@a��s33�aG����C�q�@a��l(��n{�%C��q                                    Bx~�H  �          @��H@�\)�<��?�G�A[�C���@�\)�H��?+�@�G�C��3                                    Bx~��  �          @���@�Q��*=q?У�A��C��@�Q��:�H?��A>�RC�Ǯ                                    Bx~�  �          @�=q@�
=�<��?���Af{C��H@�
=�I��?:�H@���C���                                    Bx~:  �          @��H@���G
=?�  AX(�C���@���R�\?!G�@�Q�C���                                    Bx~�  �          @��H@���K�?��
A2{C�Q�@���Tz�>���@�G�C���                                    Bx~�  �          @�33@�Q��X��?��\A0z�C���@�Q��a�>�33@p  C�W
                                    Bx~-,  �          @�33@b�\�{�?.{@��C�f@b�\�\)��Q쿂�\C���                                    Bx~;�  �          @��@tz��l��?.{@陚C��R@tz��p�׽L�Ϳ�C���                                    Bx~Jx  �          @��
@p���mp�?Y��A�C��@p���s�
=�G�?�
=C�Q�                                    Bx~Y  T          @�=q@fff�qG�?��A8  C�ٚ@fff�y��>���@Q�C�Y�                                    Bx~g�  �          @��\@y���a�?Y��A{C��3@y���g�>\)?�  C���                                    Bx~vj  �          @���@tz��aG�?��
A4(�C���@tz��j=q>��
@_\)C��                                    Bx~�  �          @�
=@s33�Y��?�ffA:{C��@s33�b�\>�Q�@~{C���                                    Bx~��  �          @��@vff�Vff?�(�A��\C���@vff�dz�?G�A33C��
                                    Bx~�\  �          @�G�@z�H�[�?aG�A�C�o\@z�H�a�>8Q�?�p�C�H                                    Bx~�  �          @�G�@p  �c�
?���A;
=C�C�@p  �l��>�{@n{C���                                    Bx~��  �          @��@w
=�c�
?��A3�C��@w
=�l��>��R@VffC�)                                    Bx~�N  T          @�z�@vff�g
=?�{A=p�C�p�@vff�p  >�Q�@w�C��)                                    Bx~��  �          @�p�@^{�r�\?˅A��C�9�@^{��G�?L��A��C�XR                                    Bx~�  �          @��@P  �qG�@�
A��
C�aH@P  ���?�G�AXz�C�.                                    Bx~�@  �          @�{@Q��g�@
=A��C�q@Q�����?�=qA�p�C��H                                    Bx~ �  �          @��@W
=�|��?�  A���C�/\@W
=��p�?+�@�C�k�                                    Bx~ �  �          @��@U��Q�?�AIC�޸@U��p�>���@b�\C�XR                                    Bx~ &2  �          @�(�@U��s�
?�\A�  C��@U����H?uA$Q�C��                                    Bx~ 4�  �          @�(�@Z=q�s33?�33A���C��{@Z=q���?W
=A�
C��                                    Bx~ C~  �          @��
@^�R�s33?�p�A�Q�C�@ @^�R����?+�@�
=C�p�                                    Bx~ R$  �          @�@j=q�xQ�?}p�A'�C���@j=q��  >.{?�ffC�<)                                    Bx~ `�  T          @��H@_\)�~�R?.{@陚C���@_\)���þ����33C�q�                                    Bx~ op  �          @�(�@p  �tz�?   @��\C�:�@p  �u�����L(�C�'�                                    Bx~ ~  �          @��
@w��l��>��@0��C�+�@w��j�H��\��\)C�E                                    Bx~ ��  �          @�=q@u��k�>�\)@@  C�\@u��j=q���H����C�%                                    Bx~ �b  �          @��@w��k�>��@��
C�9�@w��k���p��z�HC�5�                                    Bx~ �  �          @�33@s�
�l(�?
=@��HC��3@s�
�o\)�B�\��C��=                                    Bx~ ��  �          @�33@qG��k�?s33A#�C��R@qG��s33>#�
?ٙ�C�e                                    Bx~ �T  �          @���@xQ��`��?O\)A��C��@xQ��fff=u?��C��                                    Bx~ ��  �          @���@x���_\)?fffA��C�
=@x���fff>��?˅C��R                                    Bx~ �  �          @��@s33�e�?s33A%G�C�Z�@s33�l��>.{?��C��H                                    Bx~ �F  �          @�(�@w��j=q?:�H@�Q�C�L�@w��n�R��\)�0��C�f                                    Bx~!�  �          @��H@w
=�g
=?8Q�@���C�xR@w
=�k��u�&ffC�0�                                    Bx~!�  �          @���@q��hQ�?0��@�  C��@q��l(����Ϳ��C���                                    Bx~!8  �          @�\)@fff�p��>��H@��
C�� @fff�q녾��
�c�
C��                                    Bx~!-�  �          @��H@e��z=q?&ff@޸RC�:�@e��}p��W
=���C�\                                    Bx~!<�  �          @�=q@qG��mp�>��H@��C���@qG��n�R���
�^�RC���                                    Bx~!K*  �          @��@z�H�]p�>k�@!G�C�N@z�H�Z�H����{C�o\                                    Bx~!Y�  �          @��\@h���vff?\)@���C��\@h���x�þ�\)�C33C���                                    Bx~!hv  �          @�33@[����>\@��
C�R@[�������\��ffC�%                                    Bx~!w  �          @���@P  ��z�=��
?c�
C�@P  ��녿Q��33C�T{                                    Bx~!��  �          @�=q@W����?&ff@߮C�� @W���33��  �.{C���                                    Bx~!�h  �          @�p�@O\)��=q?�ffA���C�G�@O\)���?�R@��HC�z�                                    Bx~!�  �          @�ff@k��u?�z�AE�C��H@k���  >�z�@Dz�C�J=                                    Bx~!��  �          @�\)@g
=�z�H?��\AV�\C�S3@g
=��33>\@}p�C���                                    Bx~!�Z  �          @�
=@l(��u�?��RAQG�C��{@l(���Q�>�Q�@r�\C�N                                    Bx~!�   �          @��@n{�qG�?�=qA8��C�U�@n{�z=q>aG�@Q�C��=                                    Bx~!ݦ  �          @��
@s33�n{?=p�@��RC�Ǯ@s33�r�\���Ϳ���C��H                                    Bx~!�L  �          @�=q@h���tz�?�R@�
=C�˅@h���w
=�u�'�C���                                    Bx~!��  �          @��R@1����@��A�
=C�� @1�����?�z�ADQ�C���                                    Bx~"	�  �          @�@S�
��=q?���Ab{C��3@S�
��Q�>\@�  C��                                    Bx~">  �          @��R@U���\?�\)Aip�C�� @U��G�>�
=@�{C���                                    Bx~"&�  �          @�\)@U����H?�p�Az�HC���@U����?�@��C��R                                    Bx~"5�  �          @�{@Mp���(�?��RA~{C��=@Mp����?�@��RC�+�                                    Bx~"D0  �          @�z�@N{��=q?���Ay�C�*=@N{��G�>��H@�{C�n                                    Bx~"R�  �          @��@K����
?�{Aj�RC���@K���=q>Ǯ@�{C�.                                    Bx~"a|  �          @�ff@QG����?s33A ��C��R@QG����H�#�
���C��H                                    Bx~"p"  �          @��R@e�u�?��RA|��C��{@e��=q?�@���C��R                                    Bx~"~�  "          @���@Q����H?��
A[�C�W
@Q�����>��R@R�\C���                                    Bx~"�n  �          @��R@W
=��G�?��RA~=qC��@W
=����?�@���C�
                                    Bx~"�  �          @�@R�\���
?�  AUG�C�G�@R�\����>�=q@4z�C��3                                    Bx~"��  �          @�(�@S�
���H?�{A?
=C�|)@S�
���>\)?�(�C���                                    Bx~"�`  �          @��@S33��(�?O\)A�
C�O\@S33���R�.{��\C�                                    Bx~"�  �          @��H@N{��{?:�H@��C�˅@N{�������1G�C��)                                    Bx~"֬  �          @��\@U��z�>�33@p  C�q�@U��33��R��\)C��\                                    Bx~"�R  �          @��
@L(���
=?k�Ap�C���@L(���녽�Q�}p�C�<)                                    Bx~"��  �          @�(�@HQ�����?s33A"�\C�#�@HQ����
���
�Q�C��\                                    Bx~#�  �          @��@E���\?(�@У�C��=@E��33��
=����C��R                                    Bx~#D  �          @��\@E��G�?333@��HC��H@E���\�����`  C��)                                    Bx~#�  �          @���@J�H��{?!G�@��C��{@J�H��
=�\���C�z�                                    Bx~#.�  �          @�=q@I�����R?5@�C�` @I����Q쾞�R�VffC�8R                                    Bx~#=6  �          @��@P  ���?E�A{C�,�@P  ���k����C��3                                    Bx~#K�  �          @�33@QG���p�?5@�z�C��@QG���
=�����P��C�޸                                    Bx~#Z�  �          @�(�@QG���\)?z�@�z�C��H@QG������ff��=qC��{                                    Bx~#i(  �          @��@N{��\)?(�@��HC�� @N{��Q��
=��\)C���                                    Bx~#w�  �          @��@K���Q�?333@�\)C�g�@K�������33�l��C�C�                                    Bx~#�t  �          @��H@L����\)?#�
@�(�C���@L����Q��������C�y�                                    Bx~#�  �          @�33@\(�����?8Q�@���C�AH@\(����\��=q�:=qC��                                    Bx~#��  �          @��\@J=q��ff?B�\A�C�~�@J=q��Q쾏\)�A�C�N                                    Bx~#�f  �          @�(�@<����ff?+�@��
C�Ǯ@<����\)��(���(�C���                                    Bx~#�  �          @��@<����{>��H@���C�Ф@<����p�����θRC���                                    Bx~#ϲ  �          @��@@  ����?L��A33C��@@  ���H��=q�;�C�P�                                    Bx~#�X  �          @�p�@���L��?��A�z�C�޸@���`  ?L��A  C��R                                    Bx~#��  �          @��@��R�"�\?���A�Q�C�.@��R�<(�?�G�AT��C�N                                    Bx~#��  T          @�ff@�G��/\)?�Q�A���C���@�G��HQ�?���AK�
C��\                                    Bx~$
J  T          @�{@�=q�#33@ffA��C��{@�=q�?\)?�33An�HC���                                    Bx~$�  �          @�33@�
=�(�@p�A���C��)@�
=�:=q?��
A�G�C��{                                    Bx~$'�  �          @��@�(��'�@�A�=qC��\@�(��C33?�{Am��C�                                    Bx~$6<  �          @��\@����>�R?�A�\)C���@����U?�G�A/�C�&f                                    Bx~$D�  �          @��\@s33�S�
?�p�A��
C�y�@s33�hQ�?Y��A�HC�                                      Bx~$S�  �          @��\@p���\(�?\A���C���@p���n{?(�@љ�C��=                                    Bx~$b.  �          @��H@l(��fff?�{Alz�C��)@l(��u�>��@�p�C��)                                    Bx~$p�  �          @�G�@���!�?�A���C��R@���:=q?�{AC\)C��                                    Bx~$z  �          @���@�=q��?�33A���C���@�=q�1G�?��RAY��C��                                    Bx~$�   �          @�G�@��H��?�\)A��C���@��H�1G�?��HAS�C���                                    Bx~$��  �          @��@��
���?�A�C���@��
�2�\?��AF�RC��)                                    Bx~$�l  �          @���@��H�(�?��A�Q�C�Z�@��H�3�
?�{AA�C���                                    Bx~$�  �          @���@�(����?�(�A��C��\@�(��0  ?�ffA7\)C��\                                    Bx~$ȸ  �          @��H@��
�   ?�G�A���C��@��
�7�?�ffA6=qC�\)                                    Bx~$�^  �          @��\@���{?�
=A��C�` @���4z�?}p�A+33C��{                                    Bx~$�  �          @�=q@�ff�=q?�33A�=qC��@�ff�0  ?xQ�A((�C�&f                                    Bx~$��  �          @�G�@�  �  ?�{A�{C��{@�  �%?xQ�A)�C��                                    Bx~%P  �          @�G�@�
=��\?��A�Q�C�u�@�
=�(Q�?}p�A,Q�C�                                    Bx~%�  �          @���@�z���?�\)A�
=C��H@�z��1G�?n{A"�RC��                                     Bx~% �  �          @�
=@�����R?�A���C��=@����5�?xQ�A+
=C�8R                                    Bx~%/B  �          @��R@����   ?�\)A�p�C���@����5?k�A!�C�4{                                    Bx~%=�  �          @�ff@����   ?���A�
=C���@����5�?aG�A��C�8R                                    Bx~%L�  �          @�ff@����!�?�ffA��HC��\@����6ff?Tz�A\)C�&f                                    Bx~%[4  �          @�  @����'
=?�ffA�C�XR@����:�H?O\)A=qC��R                                    Bx~%i�  �          @�{@�  �#33?��A�C��f@�  �7
=?O\)A\)C��                                    Bx~%x�  �          @�  @����?�z�A��C��@����,(�?z�HA,z�C�B�                                    Bx~%�&  �          @�Q�@��H�8Q�?�@�G�C�:�@��H�;��W
=��
C�H                                    Bx~%��  �          @�Q�@�ff�H��>�?�33C���@�ff�C�
�5���C��q                                    Bx~%�r  �          @���@�
=�E?B�\A�C��H@�
=�K����Ϳ��C�}q                                    Bx~%�  �          @���@����<(�?�  A.{C��
@����Fff>.{?�\)C�H                                    Bx~%��  �          @�=q@�\)�A�?�=qA;�
C�5�@�\)�Mp�>k�@p�C�l�                                    Bx~%�d  �          @�33@��R�C�
?�z�AI��C��q@��R�P��>���@K�C�q                                    Bx~%�
  T          @�(�@�=q�@��?���A<(�C���@�=q�L��>u@$z�C���                                    Bx~%��  �          @��
@����>{?�(�AR�\C���@����L(�>�p�@~�RC��q                                    Bx~%�V  �          @��@��H�:=q?�AI�C��@��H�G�>���@b�\C�&f                                    Bx~&
�  �          @��\@�\)�&ff?��RAW33C��@�\)�5>�@�
=C��3                                    Bx~&�  �          @�=q@����?�G�A\  C��R@���+�?��@��RC�                                    Bx~&(H  �          @�33@�z��333?��HAP��C���@�z��AG�>���@�  C��                                     Bx~&6�  �          @�=q@��H�1G�?�=qAhQ�C���@��H�A�?�@�(�C��=                                    Bx~&E�  �          @��@���<(�?�p�AW
=C��H@���J�H>\@�=qC��                                     Bx~&T:  �          @�33@��-p�?��A_�
C�E@��=p�?   @��C�q                                    Bx~&b�  �          @�=q@���0��?�ffAb{C���@���@��>�@�
=C���                                    Bx~&q�  �          @��@�(��.{?�ffAb{C��@�(��>{>��H@�=qC��\                                    Bx~&�,  T          @��\@���*�H?��Ad��C�c�@���;�?�@��\C�/\                                    Bx~&��  T          @�=q@����+�?�ffAb=qC�Z�@����;�?   @���C�+�                                    Bx~&�x  T          @��@��\�/\)?�33At��C��q@��\�AG�?z�@���C���                                    Bx~&�  �          @�  @�p��C�
?W
=A\)C��)@�p��K��L�Ϳ
=C�]q                                    Bx~&��  �          @�Q�@����C�
?�ffA7�
C�Ǯ@����N�R>\)?�=qC��                                    Bx~&�j  �          @���@�G��S�
?5@�\)C�\)@�G��W���  �.�RC��                                    Bx~&�  �          @��@����4z�?���A;�
C�J=@����@��>aG�@��C�p�                                    Bx~&�  �          @��@�Q��3�
?�
=APQ�C�N@�Q��A�>�{@k�C�L�                                    Bx~&�\  �          @���@���C33?�ffA733C�� @���N{>\)?�p�C��                                    Bx~'  T          @�  @�\)�3�
?��Ag\)C�7
@�\)�C�
>�@��C�
=                                    Bx~'�  "          @���@�\)�;�?�AL��C��@�\)�H��>�\)@E�C��3                                    Bx~'!N  �          @���@��\�5?���A@(�C�Y�@��\�B�\>u@'
=C�w
                                    Bx~'/�  �          @��\@���;�?���A:{C��f@���G
=>B�\@ ��C�3                                    Bx~'>�  �          @��\@�=q�@��?fffA�HC��
@�=q�H�ü#�
�uC��                                    Bx~'M@  �          @���@�p��B�\?��A>ffC��R@�p��N{>#�
?��C�(�                                    Bx~'[�  �          @�Q�@���@��?�ffAdQ�C�޸@���P��>�p�@���C���                                    Bx~'j�  �          @�Q�@�Q��:=q?�ffA8��C��R@�Q��E>.{?�C��                                    Bx~'y2  �          @���@�
=�?\)?��A9C�T{@�
=�J�H>��?���C���                                    Bx~'��  �          @�G�@����;�?uA&�RC��)@����E�=u?+�C�/\                                    Bx~'�~  �          @���@�  �<��?xQ�A*�\C��R@�  �G
==�\)?@  C��f                                    Bx~'�$  �          @�=q@�
=�Fff?xQ�A'�C�ٚ@�
=�P  <��
>B�\C�5�                                    Bx~'��  �          @�33@�{�E?�p�AT  C���@�{�Tz�>��@5�C��{                                    Bx~'�p  �          @�=q@���A�?�=qA:�RC�5�@���Mp�>\)?��HC�h�                                    Bx~'�  T          @���@�G��3�
?�AL��C�g�@�G��A�>�z�@G
=C�g�                                    Bx~'߼  �          @�=q@���5�?���AD��C��f@���A�>�  @*=qC��{                                    Bx~'�b  T          @��@�p��1G�?�G�A0(�C��\@�p��<��>��?�=qC�!H                                    Bx~'�  �          @��\@���(��?B�\A
=C��@���0  �u�+�C�s3                                    Bx~(�  �          @�=q@�Q��<(�?��HAR�RC��3@�Q��J=q>�\)@AG�C���                                    Bx~(T  �          @�(�@�p��@��?ǮA�z�C�
@�p��U?z�@�\)C��=                                    Bx~((�  �          @��H@�33�6ff?�33A���C��q@�33�Q�?uA%C��                                    Bx~(7�  �          @�G�@~{�9��?�Q�A�  C��@~{�Vff?z�HA*ffC��\                                    Bx~(FF  �          @�G�@x���<��@33A�{C�o\@x���[�?���A9�C�Q�                                    Bx~(T�  �          @��@�G��8��@   A�p�C�4{@�G��W
=?��
A2=qC�                                      Bx~(c�  �          @�@|(��H��@�A�p�C��{@|(��g
=?z�HA&�HC��                                     Bx~(r8  �          @�(�@z=q�HQ�?�A�{C���@z=q�dz�?aG�A�C���                                    Bx~(��  �          @�=q@}p��=p�@   A�  C��q@}p��[�?�  A-��C���                                    Bx~(��  �          @��
@����A�?��A��C���@����\��?^�RA��C���                                    Bx~(�*  �          @���@�  �?\)@33A��RC��q@�  �^{?��A1�C���                                    Bx~(��  �          @��\@~�R�:�H@   A�ffC��q@~�R�Y��?�G�A.�RC�Ǯ                                    Bx~(�v  �          @���@w��C33?��A���C��@w��^�R?Y��A��C�f                                    Bx~(�  �          @���@s�
�C�
?�p�A��\C��q@s�
�aG�?p��A$��C��H                                    Bx~(��  �          @��@w��AG�@�A�{C��@w��`  ?�  A.=qC���                                    Bx~(�h  T          @���@�  �E�?�z�A�=qC�4{@�  �aG�?^�RAQ�C�S3                                    Bx~(�  �          @���@|���@��?��
A�\)C�` @|���Z=q?B�\A�
C��q                                    Bx~)�  �          @�Q�@{��1�@
=A�p�C�Z�@{��S33?��AH��C��q                                    Bx~)Z  �          @�Q�@~{�1�@�A��C���@~{�QG�?��A:ffC�B�                                    Bx~)"   �          @��
@���7
=@G�A��C�h�@���Vff?��
A0��C�=q                                    Bx~)0�  �          @�
=@�p��?\)?��A�  C�0�@�p��[�?\(�Ap�C�G�                                    Bx~)?L  �          @�@��H�J=q?�A�33C�(�@��H�aG�?
=@�G�C��H                                    Bx~)M�  �          @�{@�  �U�?�=qA��C��@�  �j=q>�ff@���C��H                                    Bx~)\�  �          @�
=@�G��W
=?�G�A�(�C�
@�G��j�H>�p�@y��C��q                                    Bx~)k>  �          @�(�@���N�R?�
=Av�HC��H@���`��>�{@g�C��\                                    Bx~)y�  �          @��@����C�
?�Q�Ayp�C�Ǯ@����W
=>���@��C�~�                                    Bx~)��  �          @�Q�@��\�A�?���As\)C��R@��\�S�
>�33@s�
C�|)                                    Bx~)�0  T          @���@�=q�G�?�Q�APz�C�AH@�=q�U>�?���C�S3                                    Bx~)��  T          @�Q�@��\�AG�?���An�\C���@��\�S33>��
@^�RC���                                    Bx~)�|  �          @���@�(��B�\?���Ak�C��3@�(��S�
>��R@S�
C��                                    Bx~)�"  �          @�z�@�z��I��?�33Ap��C�Y�@�z��[�>��
@W
=C�'�                                    Bx~)��  �          @��H@����C�
?�Q�A��
C�h�@����\(�?�R@�C���                                    Bx~)�n  �          @��H@�=q�AG�?�
=A���C���@�=q�Y��?�R@ָRC�3                                    Bx~)�  �          @�=q@��
�8Q�?�G�A�ffC�� @��
�S33?=p�@��RC���                                    Bx~)��  �          @��@����<��?�  A�{C���@����W
=?333@�C�&f                                    Bx~*`  �          @���@�Q��Fff?��A���C�#�@�Q��b�\?E�A�
C�C�                                    Bx~*  T          @�z�@����<(�@ ��A�\)C��@����[�?s33A"�\C�ٚ                                    Bx~*)�  �          @�(�@��H�333@
=A�Q�C���@��H�U�?�=qA9��C�g�                                    Bx~*8R  �          @��H@��H�.�R@�
A�C��@��H�P  ?��A6�RC��q                                    Bx~*F�  �          @�z�@��R�1�?��A��C�E@��R�P  ?^�RA��C�0�                                    Bx~*U�  �          @�p�@��R�4z�?�z�A���C�{@��R�R�\?aG�AG�C��q                                    Bx~*dD  �          @��@~�R�.{@(�A�z�C���@~�R�W�?�z�Aq�C��                                     Bx~*r�  �          @�p�@�(��1G�@��A���C��@�(��Tz�?�\)A>=qC���                                    Bx~*��  �          @��@�z��1G�@�A�(�C�)@�z��S�
?��A9G�C��=                                    Bx~*�6  �          @��@�ff�1�?�p�A��C�<)@�ff�Q�?s33A!C�H                                    Bx~*��  �          @�@����.{?�Q�A��C���@����L��?n{Ap�C��
                                    Bx~*��  �          @��H@�ff�0��?�A�Q�C�T{@�ff�Mp�?J=qA(�C�Q�                                    Bx~*�(  �          @��@���\)@   A���C�˅@���@��?��A8  C�N                                    Bx~*��  �          @��H@�ff� ��@�A�
=C���@�ff�E�?�z�AH��C��                                    Bx~*�t  �          @���@~{�333@z�A�(�C�n@~{�Z=q?�  AW\)C��                                    Bx~*�  �          @�=q@y���*=q@�A�  C�޸@y���Tz�?�33At��C�Ф                                    Bx~*��  �          @��
@~{�G
=?�=qA�p�C��\@~{�c33?.{@��C��                                    Bx~+f  �          @��
@xQ��X��?��RA���C�q�@xQ��l(�>��@3�
C�9�                                    Bx~+  �          @���@z=q�B�\?���A�p�C��@z=q�^{?0��@��C�.                                    Bx~+"�  T          @�  @{��;�?�{A�{C���@{��X��?B�\Ap�C��                                     Bx~+1X  �          @�z�@��H�{@ffA�C�n@��H�B�\?��AI�C���                                    Bx~+?�  �          @�33@|���W�?��A_
=C��f@|���g
==�\)?E�C��                                    Bx~+N�  �          @�33@����Q�?��\A\��C�]q@����aG�=��
?c�
C�aH                                    Bx~+]J  �          @��@�=q�G
=?�\)Ao�C�U�@�=q�X��>k�@   C�!H                                    Bx~+k�  �          @���@~{�Mp�?���Ar�\C��f@~{�^�R>W
=@\)C�Y�                                    Bx~+z�  �          @��\@~�R�J�H?���A��C���@~�R�aG�>Ǯ@�
=C�E                                    Bx~+�<  �          @��
@x���aG�?�33AF�RC��\@x���l�ͽ����\C�5�                                    Bx~+��  �          @�(�@����U?�33AE��C�=q@����a녽�\)�333C�s3                                    Bx~+��  T          @��
@����U?���A=p�C�>�@����`�׽�G���(�C��f                                    Bx~+�.  T          @��@|���Z�H?c�
A=qC��=@|���aG���{�n�RC�&f                                    Bx~+��  �          @�ff@U�~�R=#�
>�G�C��)@U�p  ����n�\C��{                                    Bx~+�z  �          @��@O\)����#�
����C�
@O\)�vff���H��G�C��                                    Bx~+�   �          @��@G���ff�#�
��C�P�@G��|(���(���{C�5�                                    Bx~+��  �          @�\)@_\)�xQ�>���@P��C���@_\)�p  �����<��C�~�                                    Bx~+�l  �          @��@tz��dz�>�G�@��\C�|)@tz��`  �Tz����C���                                    Bx~,  �          @���@�{�I��?@  A�\C��f@�{�N{�Ǯ��Q�C�>�                                    Bx~,�  �          @��@�  �E?B�\A33C�  @�  �J=q��p��\)C���                                    Bx~,*^  �          @�G�@�{�'
=?�z�AK�C�˅@�{�6ff>8Q�?�C��=                                    Bx~,9  �          @���@�
=�"�\?�(�AUC�33@�
=�333>�  @0  C��                                    Bx~,G�  �          @�G�@�=q�5�?��A5C�^�@�=q�AG����
��  C���                                    Bx~,VP  �          @�33@l(��r�\?=p�@�\)C�#�@l(��s�
�#�
�ۅC��                                    Bx~,d�  �          @��@���7
=?^�RA�RC�c�@���>�R�L�����C���                                    Bx~,s�  �          @�33@�Q��,(�?}p�A+�C���@�Q��7����
�W
=C�Ǯ                                    Bx~,�B  T          @�33@����,��?^�RA�\C��R@����5�\)��p�C���                                    Bx~,��  �          @�=q@�p����?8Q�@�  C�(�@�p��"�\�L���Q�C��{                                    Bx~,��  �          @�G�@�ff�0  ?@  AffC�#�@�ff�5��\)�@��C��q                                    Bx~,�4  �          @���@����1�?c�
A33C�ٚ@����:=q�#�
��(�C�:�                                    Bx~,��  �          @��@���H��?E�A�C�Q�@���Mp�������(�C�f                                    Bx~,ˀ  �          @�G�@�=q�S�
?�R@�
=C�q�@�=q�S�
�(��ҏ\C�o\                                    Bx~,�&  �          @�G�@��\�U�?
=q@��\C�c�@��\�S33�0����  C���                                    Bx~,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx~,�r   �          @���@�33�Q�?
=@�ffC���@�33�QG���R��G�C��3                                    Bx~-  �          @�=q@�{�N�R>�(�@�C�4{@�{�J�H�B�\�\)C�xR                                    Bx~-�  "          @�=q@�
=�L��>�(�@�z�C�q�@�
=�H�ÿ@  �{C��{                                    Bx~-#d  �          @��\@�p��P  ?5@�ffC��@�p��Q녿�\��ffC��H                                    Bx~-2
  T          @�(�@�{�Q�?B�\A��C��R@�{�U�����z�C��H                                    Bx~-@�  �          @��\@�(��Q�?333@�C���@�(��S�
����  C��H                                    Bx~-OV  �          @�G�@���I��?W
=A�HC�h�@���O\)��Q��z=qC��                                    Bx~-]�  �          @��@���C�
?c�
AC�3@���J�H��z��H��C���                                    Bx~-l�  �          @��\@���QG�?s33A$��C�� @���X�þ����QG�C�<)                                    Bx~-{H  �          @�G�@�(��I��?z�HA+
=C�T{@�(��R�\�u�%�C���                                    Bx~-��  �          @�=q@�(��L(�?��\A1��C�1�@�(��U�W
=��\C���                                    Bx~-��  �          @��\@����<(�?�p�AUG�C���@����L(�=�\)?333C���                                    Bx~-�:  �          @��H@���1�?�ffAaG�C��)@���C�
>8Q�?��RC�t{                                    Bx~-��  �          @��@���J�H?���A@��C��
@���W�����p�C��                                    Bx~-Ć  �          @�\)@�=q�H��?�p�AO33C��
@�=q�XQ켣�
�8Q�C��q                                    Bx~-�,  �          @�{@�
=�L(�?��\AW�C�z�@�
=�[�    <��
C�u�                                    Bx~-��  �          @�p�@�(��S33?�33AC�C���@�(��_\)�#�
��z�C��                                    Bx~-�x  �          @��@�=q�N{?��
A,��C���@�=q�W��u��RC��                                    Bx~-�  �          @�Q�@���L(�?�ffA/�C���@���Vff�W
=�p�C�7
                                    Bx~.�  �          @�
=@���A�?�33AB�HC�@���O\)�L�Ϳ\)C��R                                    Bx~.j  �          @�
=@�p��AG�?�{A;\)C���@�p��Mp����Ϳ�  C��)                                    Bx~.+  �          @��@�Q��<(�?�\)A=�C�u�@�Q��H�ýL�Ϳ
=qC���                                    Bx~.9�  �          @�
=@�{�AG�?���A4  C�� @�{�L�;\)����C�R                                    Bx~.H\  �          @�  @�\)�=p�?�
=AE�C�C�@�\)�L(����
�.{C�H�                                    Bx~.W  �          @��@�Q��5?��A]G�C���@�Q��HQ�>��?���C���                                    Bx~.e�  �          @�  @����333?��Ab=qC�'�@����Fff>B�\@ ��C��3                                    Bx~.tN  �          @�  @����3�
?��AX(�C�.@����E�>\)?�z�C��3                                    Bx~.��  �          @���@����7�?�p�AN{C���@����G�=u?z�C��                                     Bx~.��  �          @�  @����5�?�ffA[�C��@����G
=>\)?�(�C��H                                    Bx~.�@  T          @�Q�@�Q��=p�?�33A?�
C�Z�@�Q��J�H�u�(�C�n                                    Bx~.��  T          @���@�p��G�?���A2�RC�h�@�p��R�\�B�\���RC��                                    Bx~.��  �          @�\)@�=q�N{?k�A�C���@�=q�U���p��w
=C�.                                    Bx~.�2  �          @�ff@����g�?&ff@ٙ�C��q@����e�G���C�R                                    Bx~.��  �          @�p�@xQ��n�R?\)@�z�C��@xQ��j=q�h���ffC�P�                                    Bx~.�~  �          @��R@{��p��>�G�@�z�C�#�@{��h�ÿ��\�-�C��
                                    Bx~.�$  �          @�Q�@����mp�?#�
@��
C���@����j�H�Tz��
�\C��\                                    Bx~/�  �          @�{@�{�S�
?��
A/�C���@�{�]p����R�Q�C�9�                                    Bx~/p  �          @�  @����g�?uA Q�C��@����mp��   ��  C��=                                    Bx~/$  �          @�Q�@}p��n�R?fffA33C�\)@}p��r�\�
=��C�                                      Bx~/2�  �          @���@���fff?xQ�A"�\C�8R@���l�;��H����C��
                                    Bx~/Ab  �          @�\)@�G��aG�?��A8z�C�}q@�G��j�H��{�g
=C��                                    Bx~/P  T          @�Q�@��]p�?uA ��C�8R@��c�
��G����
C��                                    Bx~/^�  �          @��@��R�U�?���A>{C��{@��R�`�׾u�#33C��                                    Bx~/mT  �          @�\)@�ff�Q�?��\AV�RC�  @�ff�aG����Ϳ�  C�f                                    Bx~/{�  �          @�ff@�ff�QG�?�
=AHz�C�
=@�ff�^{�8Q����C�33                                    Bx~/��  �          @�ff@�  �K�?�(�AN�HC���@�  �Z=q��G���z�C���                                    Bx~/�F  �          @��R@���Mp�?�p�APz�C�ff@���\(���G����HC�u�                                    Bx~/��  �          @�@�z��U�?���A;
=C��3@�z��`  ��\)�?\)C��                                    Bx~/��  �          @�z�@�{�S33?:�H@�  C��)@�{�U���R��33C���                                    Bx~/�8  �          @��@�=q�G�?n{A�HC�\@�=q�O\)��Q��s�
C��                                    Bx~/��  �          @��R@�33�K�?Tz�A�
C��@�33�P  ������C���                                    Bx~/�  �          @�\)@�Q��5�?��\AW33C���@�Q��Fff=u?+�C��q                                    Bx~/�*  �          @��R@�G��:�H?O\)A��C��@�G��@  ��������C�E                                    Bx~/��  �          @���@�33�@  ?(�@ə�C�t{@�33�?\)��R��C�w
                                    Bx~0v  �          @�Q�@���5?s33A�
C�.@���?\)�u�\)C���                                    Bx~0  �          @�G�@���@  ?c�
A\)C�P�@���G
=��p��tz�C���                                    Bx~0+�  �          @�Q�@�(��6ff?aG�A�\C�+�@�(��>{���
�Tz�C���                                    Bx~0:h  �          @�  @���,��?��A7�C���@���:=q�u�!G�C�                                      Bx~0I  �          @��R@��\�2�\?}p�A'33C�P�@��\�<�;B�\�33C��{                                    Bx~0W�  �          @�\)@����9��?��\A,z�C���@����Dz�W
=�(�C��                                    Bx~0fZ  �          @�
=@���B�\?5@�RC���@���Dz������C��                                    Bx~0u   �          @��@�\)�Fff?&ff@�G�C��@�\)�Fff�!G���z�C��=                                    Bx~0��  �          @��@�{�L(�?�@�ffC�%@�{�H�ÿJ=q�Q�C�aH                                    Bx~0�L  �          @�G�@��QG�?&ff@�ffC���@��P�׿5���HC���                                    Bx~0��  �          @���@�z��R�\?z�@�=qC��R@�z��P  �G��G�C���                                    Bx~0��  �          @�Q�@�33�L(�?��A.�\C��@�33�U���
�U�C�AH                                    Bx~0�>  �          @��@�  �K�?��Aj�RC��
@�  �^{    <�C�`                                     Bx~0��  �          @�  @����HQ�?�
=AqC��@����\(�=u?(�C���                                    Bx~0ۊ  �          @�G�@�=q�I��?�
=Ap  C���@�=q�]p�=L��?   C��H                                    Bx~0�0  �          @�G�@�33�E�?�  A{\)C�W
@�33�[�>\)?�C��                                    Bx~0��  T          @�=q@�33�AG�?��HA���C���@�33�]p�>�33@fffC��f                                    Bx~1|  �          @��@�  �G�?�\A�z�C��@�  �c�
>�p�@q�C��                                    Bx~1"  �          @���@����A�?�Q�A�(�C�b�@����]p�>��
@Tz�C��
                                    Bx~1$�  �          @��
@�{�b�\?�p�AJ�\C��@�{�n�R�����B�\C�+�                                    Bx~13n  �          @���@��H�Z=q?�@�z�C��@��H�Tz�fff���C�E                                    Bx~1B  �          @��\@����aG�?�\@�ffC�U�@����[��u��HC��R                                    Bx~1P�  T          @��H@��R�S�
?333@�C���@��R�S�
�333��{C���                                    Bx~1_`  "          @��@�
=�QG�?Tz�A��C���@�
=�U�����G�C���                                    Bx~1n  �          @�=q@���Tz�?.{@�
=C��@���S�
�:�H��C���                                    Bx~1|�  �          @�p�@�  �Vff?W
=A��C��@�  �Z=q�
=��Q�C�l�                                    Bx~1�R  �          @��@��R�S33?Tz�A��C���@��R�W
=�z����RC��H                                    Bx~1��  �          @�(�@����P��?O\)A�C��@����S�
�
=���C��                                    Bx~1��  �          @���@���O\)?J=qA ��C�Ff@���Q녿����33C��                                    Bx~1�D  �          @�\)@�(��N{?&ff@ڏ\C��\@�(��Mp��8Q����HC��                                     Bx~1��  �          @�  @q��QG�>.{?�p�C��3@q��C33��
=�ZffC��=                                    Bx~1Ԑ  �          @���@j�H�\(�>�?\C�n@j�H�L(�����m�C�}q                                    Bx~1�6  �          @��R@n{�QG�>k�@(��C�W
@n{�E������S\)C�4{                                    Bx~1��  �          @��@s33�Vff>\)?�ffC�N@s33�G
=��  �d��C�Y�                                    Bx~2 �  �          @�{@g��Vff=L��?z�C��R@g��E���=q�z�RC�Ф                                    Bx~2(  �          @��R@k��U�>�?��HC���@k��E���  �j=qC�                                    Bx~2�  T          @���@o\)�X��>��?޸RC��@o\)�I����G��fffC���                                    Bx~2,t  �          @��@n{�S�
>aG�@"�\C�+�@n{�G
=��z��X(�C��                                    Bx~2;  �          @��@r�\�N�R>���@[�C���@r�\�C�
����DQ�C��                                    Bx~2I�  �          @�\)@i���W�>���@uC���@i���L�Ϳ���J{C�^�                                    Bx~2Xf  �          @�p�@mp��N�R>.{?��RC�|)@mp��@�׿�
=�^=qC�y�                                    Bx~2g  �          @�ff@o\)�N{>B�\@��C��q@o\)�AG���z��Y�C��\                                    Bx~2u�  �          @�G�@j�H�Z�H>�33@�G�C�� @j�H�P�׿����IC�0�                                    Bx~2�X  �          @��R@h���Vff>��R@h��C��=@h���K���{�N�\C�k�                                    Bx~2��  �          @�  @l(��W
=>��R@c33C�ٚ@l(��K���\)�N=qC��q                                    Bx~2��  �          @���@u��[�>�z�@R�\C��@u��O\)��z��P(�C��                                    Bx~2�J  �          @�=q@�  �`  �#�
���C�c�@�  �J=q��  ���C�Ф                                    Bx~2��  �          @�=q@\)�_\)�\��33C�b�@\)�AG���ff����C�q�                                    Bx~2͖  �          @�(�@Mp��5�@
�HA��HC�Q�@Mp��^�R?J=qA33C�`                                     Bx~2�<  �          @�G�@5�+�@G�B�C�T{@5�o\)?�(�A���C���                                    Bx~2��  �          @���@<(��5�@:�HB(�C��@<(��r�\?��HA��C��=                                    Bx~2��  �          @��H@333���@H��B!��C��H@333�^�R?���A��C�`                                     Bx~3.  �          @�\)@:=q�\)@>{B�C�:�@:=q�Q�?�  A�ffC���                                    Bx~3�  �          @���@S33�0��@ffA��C�\@S33�_\)?xQ�A733C��3                                    Bx~3%z  �          @��@L(��(��@(Q�B��C�<)@L(��`  ?��
Ar{C�33                                    Bx~34   �          @�p�@HQ��*�H@,��B�HC�Ǯ@HQ��c�
?���AyG�C��\                                    Bx~3B�  �          @�ff@L(��%@0��B�RC�t{@L(��`��?�z�A���C�)                                    Bx~3Ql  �          @�{@XQ��p�@�A�G�C�H@XQ��G�?c�
A.�\C��                                    Bx~3`  �          @���@W����@z�A��C��@W��E?W
=A&�\C��q                                    Bx~3n�  �          @��H@Tz��p�@   A�33C��q@Tz��Dz�?G�A(�C��R                                    Bx~3}^  �          @��@[��p�?�Q�A�z�C�,�@[��<��>��H@�C��                                    Bx~3�  �          @���@W��&ff?\A���C�,�@W��@  >�=q@\��C�,�                                    Bx~3��  �          @���@\���$z�?��HA��C���@\���<��>k�@8Q�C�                                    Bx~3�P  �          @��H@dz��$z�?���A�  C�#�@dz��8��=�G�?�{C��f                                    Bx~3��  �          @��@e�   ?�
=A���C���@e�7�>k�@:�HC��                                    Bx~3Ɯ  �          @�{@n{�(Q�?�=qAU�C�ff@n{�5�\)��C�W
                                    Bx~3�B  �          @�ff@r�\�.{?�@У�C�1�@r�\�+��+��  C�Z�                                    Bx~3��  �          @�z�@mp��0��>���@k�C���@mp��(Q�h���4��C�\)                                    Bx~3�  �          @��@p  �*�H>aG�@.{C�C�@p  � �׿p���<��C��                                    Bx~44  �          @�@xQ��%�>�Q�@��
C�,�@xQ��\)�G��C���                                    Bx~4�  �          @��\@|(���
>�33@�z�C��@|(��\)�+���\C�1�                                    Bx~4�  �          @��H@x����H>�33@�C�\@x����5�
=C�}q                                    Bx~4-&  �          @��
@x���{>W
=@&ffC��=@x���z�^�R�-��C��{                                    Bx~4;�  �          @���@|���p������
C�\@|���
�H��z��h  C��                                     Bx~4Jr  �          @��H@z�H���8Q���C�u�@z�H��
��
=�n�\C�(�                                    Bx~4Y  �          @���@s�
��H��  �G
=C��@s�
����G���p�C��3                                    Bx~4g�  �          @��@b�\� �׾�G���\)C�]q@b�\����(�����C���                                    Bx~4vd  �          @��@j�H�z�Ǯ����C��R@j�H��
=������ffC�&f                                    Bx~4�
  �          @��@j�H�\)���
��
=C�K�@j�H��녿�  ����C�ff                                    Bx~4��  �          @��@u���
��\)�l��C���@u��޸R����s
=C��=                                    Bx~4�V  �          @��@tz���
�\��  C��\@tz�ٙ���p���p�C�                                    Bx~4��  �          @�(�@x�ÿ�������=qC���@x�ÿǮ��G����
C�\                                    Bx~4��  �          @���@}p���׾�
=��C�O\@}p���G���Q��z=qC���                                    Bx~4�H  �          @��@e���
>���@��\C���@e��p��8Q��z�C�)                                    Bx~4��  �          @�{@S33��?s33AQ��C�*=@S33�#33�.{��C�/\                                    Bx~4�  �          @�{@^{�33���\�c
=C��q@^{��33�����
C���                                    Bx~4�:  T          @�ff@R�\���\����C�Z�@R�\��
=������(�C�Ǯ                                    Bx~5�  �          @�p�@Vff�*�H?�@�
=C��3@Vff�(�ÿ0����\C��                                    Bx~5�  �          @��@\)�����#�
��C��R@\)�޸R�aG��6�\C�33                                    Bx~5&,  �          @�@z��P  ?�Q�A�C��=@z��j=q=�G�?�  C�4{                                    Bx~54�  �          @�?���0��@fffBJ
=C��q?������@�\Aϙ�C��H                                    Bx~5Cx  �          @��H?�G��6ff@\(�BB�C��R?�G����?�{A��C��3                                    Bx~5R  �          @��H?Y���@��@VffB;�\C���?Y����p�?ٙ�A�(�C���                                    Bx~5`�  �          @��
?.{�G
=@VffB:{C�1�?.{��Q�?�z�A��HC��                                    Bx~5oj  �          @��\?�\)�G�@J�HB/�C��R?�\)��{?��RA�=qC�t{                                    Bx~5~  �          @���?���J�H@<(�B"Q�C�5�?����z�?�G�A��C��f                                    Bx~5��  �          @��R?��H�I��@2�\B�C�p�?��H���?���Ak
=C��                                    Bx~5�\  �          @��?����J=q@333B��C�>�?������?���Ai��C���                                    Bx~5�  �          @�
=@'��Vff?�\)A�ffC�@'��g
=�u�B�\C�f                                    Bx~5��  T          @�ff@�H�AG�@Q�A�ffC�U�@�H�i��?
=@�C��                                    Bx~5�N  �          @���@{�C�
@G�A�p�C��q@{�h��>�@�C��{                                    Bx~5��  �          @\)?�{�A�?�{A�C��3?�{�b�\>�{@�Q�C�33                                    Bx~5�  �          @�G�?�\)�>{@   B�
C�S3?�\)�qG�?p��AW�C�G�                                    Bx~5�@  �          @�=q?�z��:�H@&ffBQ�C��H?�z��p��?�ffAo\)C���                                    Bx~6�  �          @���?����G�@��B	ffC���?����s33?(��A�C�\                                    Bx~6�  �          @�Q�?�
=�I��@   A�Q�C�.?�
=�mp�>��@���C��                                    Bx~62  �          @��H?��G�@p�BQ�C�:�?��q�?(�A
=qC�N                                    Bx~6-�  �          @�Q�?�Q��U@{A���C���?�Q��~�R?�@��C���                                    Bx~6<~  �          @�G�?���R�\@�B \)C���?���|��?
=@�\)C���                                    Bx~6K$  T          @���?�\)�W�@�
A��HC���?�\)�{�>�Q�@�z�C�5�                                    Bx~6Y�  �          @�?�=q�`  ?�{A���C�o\?�=q�}p�=�?�C�O\                                    Bx~6hp  �          @�ff?��H�\��?��A�{C���?��H�{�>.{@
=C�:�                                    Bx~6w  �          @�p�?�33�[�?�Q�A�  C��?�33�tz�<#�
=��
C��{                                    Bx~6��  �          @�(�?��H�Z�H?�G�A��
C�E?��H�o\)�#�
��C�K�                                    Bx~6�b  �          @�{?�ff�g�?���Atz�C�<)?�ff�n�R�(��{C��H                                    Bx~6�  �          @�{?���^{?��A��HC�K�?���n�R��z��~�RC�t{                                    Bx~6��  �          @��R@
=�Z=q?�z�A��HC��@
=�dz����ϮC�O\                                    Bx~6�T  �          @��
@C�
�G
=��Q쿏\)C�AH@C�
�0  �������
C��                                    Bx~6��  �          @���@#33�S33?\(�A9��C���@#33�U�333��\C���                                    Bx~6ݠ  �          @��@*�H�QG�>���@�\)C��q@*�H�E���33�}�C�xR                                    Bx~6�F  �          @�\)@��Z�H>�@ҏ\C���@��Q녿�=q�n=qC�C�                                    Bx~6��  �          @�{?�(��e?fffAF�HC�e?�(��g��J=q�,��C�N                                    Bx~7	�  �          @�ff?�p��e?h��AHQ�C�t{?�p��g��G��+33C�Y�                                    Bx~78  �          @�
=?�33�i��?fffAF{C��q?�33�j�H�Q��2�\C���                                    Bx~7&�  �          @�
=?���i��?�  AZ=qC���?���mp��:�H��
C�g�                                    Bx~75�  �          @�33@ ���n{?�G�AW�C�.@ ���q녿@  �
=C��)                                    Bx~7D*  �          @�{@�
�vff?J=qA$��C��@�
�s33��G��QC�33                                    Bx~7R�  �          @�=q@
=�l��?0��A�
C�ٚ@
=�g����
�\��C��                                    Bx~7av  �          @��@�R�hQ�?(��A  C���@�R�b�\���
�]��C�q                                    Bx~7p  �          @���@ff�a�?�@���C�Ф@ff�X�ÿ�{�pz�C�XR                                    Bx~7~�  �          @���@{�^�R<��
>��C���@{�G�������HC�*=                                    Bx~7�h  �          @���@Q��b�\>8Q�@�RC��)@Q��P  �����HC�#�                                    Bx~7�  �          @���@p��^�R>�p�@���C��@p��QG���p����C�w
                                    Bx~7��  �          @�Q�@Q��^�R?�@�C�33@Q��Vff��=q�k�
C���                                    Bx~7�Z  �          @���@#�
�c�
>�\)@k�C�޸@#�
�S�
�����{C�޸                                    Bx~7�   �          @���@%��X��>���@�G�C��H@%��J�H��  ��{C���                                    Bx~7֦  �          @��@(Q��b�\=�Q�?��HC�L�@(Q��Mp���  ��\)C���                                    Bx~7�L  
�          @���@'
=�X��>8Q�@�C���@'
=�G
=��{��=qC�H                                    Bx~7��  �          @��\@#33�`  =L��?��C��@#33�I�����
��p�C�~�                                    Bx~8�  �          @�\)@   �X�þu�S33C�33@   �:�H�޸R�\C�Ff                                    Bx~8>  �          @���@ff�Mp�������C�q@ff�&ff��(����C��                                    Bx~8�  �          @��\@%�W�?�Q�A��
C��H@%�qG�    �#�
C�9�                                    Bx~8.�  �          @��\@�H�b�\?���A�G�C�/\@�H�xQ�#�
��
C��)                                    Bx~8=0  �          @�33@�R�e?��HA��
C�Q�@�R�w
=���R�w�C�Y�                                    Bx~8K�  �          @��@(���dz�?�33A�Q�C�C�@(���tzᾳ33��=qC�XR                                    Bx~8Z|  �          @�\)@1G��g
=?��A~=qC��f@1G��s33�������C��                                    Bx~8i"  �          @���@2�\�l��?�33A_�
C��=@2�\�tz�!G����
C��                                    Bx~8w�  �          @�Q�@4z�����?n{A+�C�� @4z����׿z�H�4��C��=                                    Bx~8�n  �          @�p�@,(�����?^�RA"�\C���@,(��~�R����B�RC��                                    Bx~8�  �          @�  @4z�����?Q�A�C�l�@4z��~�R����IG�C���                                    Bx~8��  T          @��@<(��}p�?J=qA��C�H�@<(��x�ÿ�=q�H��C���                                    Bx~8�`  �          @�(�@B�\��G�?^�RA�C�y�@B�\��  ��ff�<��C��                                     Bx~8�  �          @�
=@E��33?k�A"=qC��H@E��=q���
�6=qC��R                                    Bx~8Ϭ  �          @��
@C�
�~�R?��\A7\)C�@C�
��Q�c�
��
C���                                    Bx~8�R  �          @�G�@6ff�j=q?�z�A`z�C��3@6ff�r�\��R��C�xR                                    Bx~8��  �          @��@5��`  ?�Q�Al(�C�y�@5��j=q����ffC��)                                    Bx~8��  �          @�p�@<���Z=q?�(�AqC�xR@<���e��ff��=qC���                                    Bx~9
D  �          @��H?��
��G�����\)C���?��
�g
=������C�s3                                    Bx~9�  �          @�=q?�33��(���
=��G�C�Q�?�33�_\)�\)��C�=q                                    Bx~9'�  �          @�ff?�33��G��aG��/\)C���?�33�n{�������C�}q                                    Bx~966  �          @��?������
��\)�O\)C�O\?����w
=��\���
C��)                                    Bx~9D�  �          @�Q�?�=q��zᾀ  �?\)C�Q�?�=q�s33�p���
=C��)                                    Bx~9S�  
�          @���?�\��p��������C��?�\�w�������C�J=                                    Bx~9b(  
�          @��R?޸R����u�333C��?޸R�w
=����z�C�"�                                    Bx~9p�  �          @�
=?�
=�������
�k�C�R?�
=�u���(����C�ff                                    Bx~9t  �          @��?�(������aG��)��C�Q�?�(��n�R�����\)C���                                    Bx~9�  �          @�  ?��H�����\���HC�C�?��H�j=q��\��RC��                                    Bx~9��  �          @���?���G��\)��=qC�� ?��dz��(���ffC��f                                    Bx~9�f  �          @���@33��G��
=��33C�@33�c33�{���RC���                                    Bx~9�  �          @�Q�?�����  �h���/�C�S3?����W��.�R�G�C��                                    Bx~9Ȳ  �          @���@Q���Q�
=��Q�C�L�@Q��a������{C���                                    Bx~9�X  �          @���@�����R��Q����C��@���e��R��{C�R                                    Bx~9��  �          @��@���G���
=��G�C�(�@��hQ��z���p�C�'�                                    Bx~9��  �          @��@����\)����H��C�3@���i���
=q��G�C��                                    Bx~:J  T          @�p�@�����
��ff��Q�C���@���^{�G����C��{                                    Bx~:�  T          @�G�@Q��}p��!G����C�  @Q��O\)�
=����C��                                    Bx~: �  T          @��\@Q��~�R�B�\�(�C�\@Q��L����R�p�C�˅                                    Bx~:/<  �          @��\@(��}p��(�����C��H@(��N{�����{C�                                      Bx~:=�  �          @��\@G��{���R���\C���@G��N{�ff��=qC���                                    Bx~:L�  �          @��\@��z�H�\)���HC�R@��O\)��\��{C��R                                    Bx~:[.  �          @��@  �y���(���{C���@  �J�H����z�C��\                                    Bx~:i�  �          @��\@���y���@  �\)C�@���HQ��(��\)C��H                                    Bx~:xz  �          @��
@\)�|�ͿO\)�"�RC��f@\)�I���!G����C��R                                    Bx~:�   �          @�
=@(��|(��^�R�+
=C�� @(��G
=�$z��=qC��                                    Bx~:��  �          @�\)@ ���z�H�W
=�$(�C�T{@ ���G
=�!����C�|)                                    Bx~:�l  �          @��@�H�xQ����=qC��R@�H�Mp��\)��z�C�y�                                    Bx~:�  �          @�\)@���\)�333�(�C�Ф@���N�R�(���(�C���                                    Bx~:��  T          @�Q�@���33�����{C�Ф@��XQ������Q�C�N                                    Bx~:�^  �          @�\)@G��������p�C�s3@G��Z�H����C��=                                    Bx~:�  �          @�
=@33��33�Ǯ��G�C���@33�^{�{��ffC���                                    Bx~:��  �          @��R@���������eC�}q@��a�������
C�n                                    Bx~:�P  �          @�{@33���\�����j=qC��{@33�`  �Q��ׅC���                                    Bx~;
�  �          @�Q�@�\�����z��\(�C�ff@�\�dz��	�����C�P�                                    Bx~;�  �          @�{@�\��=q�Ǯ���C���@�\�\(��p����\C��3                                    Bx~;(B  �          @�z�@��|(������Q�C��H@��P������=qC�H                                    Bx~;6�  �          @�(�@�}p��   ��p�C�C�@�S33�  ����C���                                    Bx~;E�  �          @�
=@����녾\���\C�=q@���\(��(��܏\C�c�                                    Bx~;T4  �          @�
=@���녾�Q����\C�&f@��\���
�H��
=C�AH                                    Bx~;b�  �          @�  @�\��z��(����RC�z�@�\�^�R�G����C��f                                    Bx~;q�  T          @��@{��z�������C��@{�Z=q����(�C��=                                    Bx~;�&  T          @�@G���녿���
=C��R@G��W��z����C���                                    Bx~;��  �          @��@   �k��8Q��Q�C�>�@   �<���z��z�C��                                    Bx~;�r  �          @�{@��fff�.{���C���@��8���  ���C���                                    Bx~;�  �          @��@�
�b�\�B�\�'\)C�{@�
�333��\�\)C�!H                                    Bx~;��  �          @���@33�c33�+��Q�C�H@33�6ff�p�� �C�ٚ                                    Bx~;�d  �          @�{@��e��
=�ffC�)@��:=q�
=q���HC��=                                    Bx~;�
  �          @�
=@��e��33��  C��f@��A녿�����Q�C��)                                    Bx~;�  �          @�\)@
�H�e��z���(�C���@
�H�:�H�	������C�K�                                    Bx~;�V  �          @�33@��_\)�aG��FffC���@��@�׿���ϙ�C��H                                    Bx~<�  �          @��H?�(��c�
��
=���
C�s3?�(��>�R�   ��  C��R                                    Bx~<�  �          @��\@ ���`�׿����  C��@ ���7������C��H                                    Bx~<!H  �          @�(�@�\�aG��0���{C�\@�\�4z��{�G�C���                                    Bx~</�  �          @�(�@z��^{�^�R�C
=C�k�@z��,(��
=�Q�C��                                    Bx~<>�  T          @��@��^�R���\�b�\C�!H@��(���   �z�C��3                                    Bx~<M:  �          @�p�@G��^�R����l��C��@G��'
=�"�\�{C��                                    Bx~<[�  �          @���@z��]p��u�V�\C�k�@z��(���(����C�f                                    Bx~<j�  �          @�p�@�_\)�h���J�RC�u�@�+�����
=C��                                    Bx~<y,  �          @�p�@ff�`  �^�R�@  C�q�@ff�.{���
�C��                                    Bx~<��  �          @�{@
=�_\)�p���O�C���@
=�+����  C�
                                    Bx~<�x  �          @�=q@�
�W��s33�YG�C��\@�
�#�
����
=C�\)                                    Bx~<�  �          @��H@{�Z=q�\����C���@{�7
=����ܣ�C��\                                    Bx~<��  �          @�=q@(��Z=q���
��ffC�aH@(��8�ÿ����HC���                                    Bx~<�j  �          @�G�@	���Y���\��ffC�+�@	���6ff����߅C���                                    Bx~<�  �          @�=q@p��Y����{���HC��H@p��7���{��G�C���                                    Bx~<߶  �          @�=q@
=q�\(��������HC��@
=q�:=q��{��\)C�C�                                    Bx~<�\  T          @�=q@
�H�Z�H�Ǯ��
=C�.@
�H�7���z��߅C���                                    Bx~<�  �          @��@
�H�Z=q������(�C�Ff@
�H�6ff��z���Q�C���                                    Bx~=�  �          @���@��R�\��
=����C�Z�@��.�R�����z�C��                                    Bx~=N  �          @��@ff�R�\��
=��{C��@ff�.�R��\)��C�U�                                    Bx~=(�  �          @���@\)�U���
=����C�H@\)�1G������  C��f                                    Bx~=7�  �          @�=q@\)�W���G�����C�Ф@\)�333��
=���
C�W
                                    Bx~=F@  �          @�=q@��Z=q��
=��\)C�O\@��5��
=����C���                                    Bx~=T�  �          @���@��\�;�G��ə�C��@��7���(���RC��{                                    Bx~=c�  �          @��@Q��[�������
=C��@Q��7�����ffC�S3                                    Bx~=r2  �          @�=q@���Y��������{C�u�@���8Q������(�C���                                    Bx~=��  �          @�  @
=q�U��Q����C�w
@
=q�3�
���ۮC��                                    Bx~=�~  �          @�  @�\�QG���z����HC���@�\�1녿޸R�θRC��H                                    Bx~=�$  �          @�  @G��[���Q����RC�B�@G��8�ÿ����33C��H                                    Bx~=��  �          @�  @ ���\(���Q���z�C�'�@ ���9��������C�`                                     Bx~=�p  T          @�Q�?�p��]p���p���33C��)?�p��:=q��z����C�{                                    Bx~=�  �          @~{?�p��Z=q�\��\)C�{?�p��7������  C�W
                                    Bx~=ؼ  �          @}p�@��W������C���@��4z��33��C��                                    Bx~=�b  �          @~�R@�
�X�þǮ���HC���@�
�5�����z�C��
                                    Bx~=�  �          @�  @��W
=�Ǯ���HC�  @��4z�����Q�C�~�                                    Bx~>�  �          @���@Q��X�þǮ����C�3@Q��5�����  C�n                                    Bx~>T  �          @~�R@���S33�\��\)C��{@���0�׿���=qC�=q                                    Bx~>!�  �          @~�R@���U������Q�C�W
@���4z�����HC��R                                    Bx~>0�  T          @���@G��^{��z���(�C��@G��=p�����33C�*=                                    Bx~>?F  �          @�  @ ���\(����
����C�.@ ���:�H������(�C�O\                                    Bx~>M�  �          @\)@G��[���\)����C�33@G��;�����  C�AH                                    Bx~>\�  �          @|(�?�
=�[���  �l��C��?�
=�<(�����׮C��)                                    Bx~>k8  �          @|(�@   �XQ쾞�R��=qC�Ff@   �7��������
C�h�                                    Bx~>y�  �          @|��?�p��Y����Q���C��?�p��7���\)��\)C�G�                                    Bx~>��  �          @|(�@   �XQ쾸Q���{C�K�@   �6ff������ffC��f                                    Bx~>�*  �          @|��?�p��XQ��ff���HC�1�?�p��3�
��Q��뙚C���                                    Bx~>��  �          @~�R?��R�Z�H��G��ʏ\C�"�?��R�5��Q���C���                                    Bx~>�v  �          @���@�\�\�;�(���p�C�Ff@�\�8Q�������
C��                                    Bx~>�  �          @~{?��R�Z�H�Ǯ���HC��?��R�7���33��(�C�Y�                                    Bx~>��  �          @���@
=�Z=q�����Q�C��q@
=�2�\����C��=                                    Bx~>�h  �          @�G�@�\�O\)�@  �,(�C�� @�\�"�\���� (�C��)                                    Bx~>�  �          @\)@z��Dzῂ�\�mC���@z��G��33�p�C��                                    Bx~>��  �          @�=q@��N{�O\)�7�C���@��   ���  C�xR                                    Bx~?Z  z          @��H@��^�R���ǮC���@��E������p�C�0�                                    Bx~?   
          @���@ ���`  =#�
?(�C��@ ���H�ÿ�ff���\C�E                                    Bx~?)�  "          @���?����n{>k�@N{C��?����[���p�����C��                                    Bx~?8L  T          @�ff?�\)�n�R>�(�@�C�S3?�\)�aG���ff����C���                                    Bx~?F�  "          @�p�@��fff>W
=@<(�C��@��S�
��Q���C�q                                    Bx~?U�  
Z          @�p�?�(��j=q>�p�@��C�'�?�(��[���=q��(�C��                                    Bx~?d>  �          @���?�
=�i��>�G�@�z�C��?�
=�]p���G�����C��\                                    Bx~?r�  �          @���?���j=q?z�@�\)C�� ?���aG�����
=C��                                    Bx~?��  "          @��@�
�fff>\)?���C���@�
�QG���  ���C�                                      Bx~?�0  "          @��@��Vff<��
>�\)C��@��@  ��  ���C��)                                    Bx~?��  �          @��@��Tz�    ���
C��\@��=p���  ���C�                                      Bx~?�|  "          @��@=q�XQ�#�
�8Q�C���@=q�@�׿����  C�XR                                    Bx~?�"  
�          @�ff@#�
�Tz����C�Ф@#�
�:=q��\)��p�C��{                                    Bx~?��  
�          @�  @%�Vff��=q�fffC���@%�7���  �¸RC�3                                    Bx~?�n  
�          @���@   �\�;.{�33C���@   �@  ���H����C��                                    Bx~?�  T          @���@(���U��  �Z=qC�#�@(���7��޸R���C�L�                                    Bx~?��  �          @�G�@.{�Tz�8Q��=qC���@.{�8Q��z����C��3                                    Bx~@`  �          @���@*=q�Vff�L���,��C�:�@*=q�9����Q���  C�Ff                                    Bx~@  "          @���@.�R�R�\�.{��\C��q@.�R�7
=�У����C���                                    Bx~@"�  "          @�
=@)���Q녾aG��>{C�o\@)���5������HC��f                                    Bx~@1R  "          @��@-p��P  �W
=�5�C���@-p��3�
��33���C��R                                    Bx~@?�  
�          @��R@(Q��QG��#�
�
�HC�j=@(Q��6ff��\)���RC�]q                                    Bx~@N�  �          @�
=@#33�Tz����(�C���@#33�.�R���H��C�|)                                    Bx~@]D  �          @�z�@,(��HQ�#�
��C�O\@,(��.�R��ff��=qC�H�                                    Bx~@k�  �          @�(�@)���J=q�.{�ffC��@)���/\)��������C��                                    Bx~@z�  T          @��H@%�I�����
��ffC���@%�*�H�ٙ���(�C�
                                    Bx~@�6  
�          @�z�@0���E�����\C���@0���,(���  ��C��
                                    Bx~@��  �          @��@5�=p�����G�C��@5�(Q쿮{��ffC��q                                    Bx~@��  �          @��H@.�R�B�\��G�����C���@.�R�*�H��(���=qC��3                                    Bx~@�(  "          @�33@*=q�G
=�.{�
=C�>�@*=q�-p���ff��p�C�=q                                    Bx~@��  �          @�@,(��L(�����eC�3@,(��.�R��z�����C�Ff                                    Bx~@�t  �          @�(�@/\)�E�������33C��=@/\)�'���33��C�R                                    Bx~@�            @��@2�\�@�׾�z���  C�g�@2�\�#�
�������C��{                                    